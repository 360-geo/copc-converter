use copc_streaming::{ByteSource, CopcError};
use reqwest::Client;

pub struct HttpSource {
    client: Client,
    url: String,
}

impl HttpSource {
    pub fn new(url: &str) -> Self {
        Self {
            client: Client::new(),
            url: url.to_string(),
        }
    }
}

/// Max concurrent range requests in `read_ranges`. R2/Cloudflare returns 503
/// when a single client fans out hundreds of range GETs at once (which a
/// deeply-paged hierarchy triggers), so cap the in-flight count.
const MAX_INFLIGHT_RANGES: usize = 8;

/// Retries per range request on transient errors (503/429/5xx/network), with
/// exponential backoff. Object-storage 503s are usually momentary.
const RANGE_RETRIES: usize = 5;

impl HttpSource {
    /// Single range GET with retry-and-backoff on transient failures.
    async fn read_range_retrying(&self, offset: u64, length: u64) -> Result<Vec<u8>, CopcError> {
        let end = offset + length - 1;
        let mut backoff_ms = 200u64;
        let mut last_err = String::new();
        for attempt in 0..=RANGE_RETRIES {
            if attempt > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(5_000);
            }
            match self
                .client
                .get(&self.url)
                .header("Range", format!("bytes={offset}-{end}"))
                .send()
                .await
            {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        match resp.bytes().await {
                            Ok(b) => return Ok(b.to_vec()),
                            Err(e) => last_err = format!("body read: {e}"),
                        }
                    } else if status.as_u16() == 429 || status.is_server_error() {
                        // Transient (incl. 503) — retry.
                        last_err = format!("HTTP {status}");
                    } else {
                        // Permanent (403/404/416/…) — don't retry.
                        return Err(CopcError::Io(std::io::Error::other(format!(
                            "HTTP {status}: range {offset}-{end}"
                        ))));
                    }
                }
                Err(e) => last_err = format!("request: {e}"),
            }
        }
        Err(CopcError::Io(std::io::Error::other(format!(
            "range {offset}-{end} failed after {RANGE_RETRIES} retries: {last_err}"
        ))))
    }
}

impl ByteSource for HttpSource {
    async fn read_range(&self, offset: u64, length: u64) -> Result<Vec<u8>, CopcError> {
        self.read_range_retrying(offset, length).await
    }

    async fn size(&self) -> Result<Option<u64>, CopcError> {
        // Use a 1-byte range GET instead of HEAD, since presigned URLs
        // are method-specific and HEAD often returns 403.
        // Parse total file size from the Content-Range header:
        //   Content-Range: bytes 0-0/123456789
        let resp = self
            .client
            .get(&self.url)
            .header("Range", "bytes=0-0")
            .send()
            .await
            .map_err(|e| CopcError::Io(std::io::Error::other(e)))?;
        Ok(resp
            .headers()
            .get("content-range")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.rsplit_once('/'))
            .and_then(|(_, total)| total.parse().ok()))
    }

    async fn read_ranges(&self, ranges: &[(u64, u64)]) -> Result<Vec<Vec<u8>>, CopcError> {
        let mut indexed: Vec<(usize, u64, u64)> = ranges
            .iter()
            .enumerate()
            .map(|(i, (o, l))| (i, *o, *l))
            .collect();
        indexed.sort_by_key(|(_, o, _)| *o);

        struct Merged {
            offset: u64,
            length: u64,
            parts: Vec<(usize, u64, u64)>,
        }
        let mut merged: Vec<Merged> = Vec::new();
        for (i, offset, length) in &indexed {
            if let Some(last) = merged.last_mut() {
                let last_end = last.offset + last.length;
                if *offset <= last_end + 1024 {
                    let new_end = (*offset + *length).max(last_end);
                    last.length = new_end - last.offset;
                    last.parts.push((*i, *offset, *length));
                    continue;
                }
            }
            merged.push(Merged {
                offset: *offset,
                length: *length,
                parts: vec![(*i, *offset, *length)],
            });
        }

        // Fetch merged ranges with bounded concurrency (not all at once) to
        // avoid overwhelming the server with a burst that triggers 503s.
        // Tag each with its index so we can reassemble in order.
        use futures::stream::StreamExt;
        let mut fetched: Vec<Vec<u8>> = vec![Vec::new(); merged.len()];
        let mut stream = futures::stream::iter(
            merged
                .iter()
                .enumerate()
                .map(|(mi, m)| async move { (mi, self.read_range(m.offset, m.length).await) }),
        )
        .buffer_unordered(MAX_INFLIGHT_RANGES);
        while let Some((mi, res)) = stream.next().await {
            fetched[mi] = res?;
        }

        let mut results = vec![Vec::new(); ranges.len()];
        for (m, data) in merged.iter().zip(fetched.iter()) {
            for (orig_idx, offset, length) in &m.parts {
                let start = (*offset - m.offset) as usize;
                let end = start + *length as usize;
                results[*orig_idx] = data[start..end].to_vec();
            }
        }
        Ok(results)
    }
}
