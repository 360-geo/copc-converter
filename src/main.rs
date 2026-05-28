use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use copc_converter::{
    Pipeline, PipelineConfig, ProgressEvent, ProgressObserver, TempCompression, collect_input_files,
};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::sync::Mutex;

// glibc's default allocator retains freed memory in per-arena caches and
// rarely returns pages to the OS, which caused monotonic RSS growth across
// the writer's batched alloc/free cycles on multi-billion-point inputs.
// mimalloc aggressively trims freed allocations back to the OS and handles
// the high-churn multi-threaded workload better for this pipeline.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Maximum fraction of the stated memory limit to actually use.
const MEMORY_SAFETY_FACTOR: f64 = 0.75;

#[derive(Parser, Debug)]
#[command(author, version, about = "Convert LAZ files to a COPC file")]
struct Args {
    /// Input LAZ/LAS file, or a directory containing them
    input: PathBuf,

    /// Output COPC file path
    output: PathBuf,

    /// Maximum memory budget (e.g. "16G", "8G", "4096M", "512M").
    /// If not specified, auto-detects from cgroup limits (K8s) or system RAM.
    #[arg(long)]
    memory_limit: Option<String>,

    /// Temp directory for intermediate files. Default: system temp
    #[arg(long)]
    temp_dir: Option<PathBuf>,

    /// Enable temporal index EVLR for GPS-time-based queries by setting the sampling stride for temporal index (every n-th point).
    /// Good value: 1000
    /// Default: None
    #[arg(long)]
    temporal_index: Option<u32>,

    /// Progress output format: "bar" (default, interactive), "plain" (log lines),
    /// or "json" (NDJSON, one JSON object per line)
    #[arg(long, value_enum, default_value_t = ProgressMode::Bar)]
    progress: ProgressMode,

    /// Maximum number of parallel threads. Default: all available cores
    #[arg(long)]
    threads: Option<usize>,

    /// Hidden: override the chunk target size in points. Bypasses the
    /// dynamic memory-budget-based calculation. Primarily for testing —
    /// force multiple chunks on a small input to exercise merge.
    #[arg(long, hide = true)]
    chunk_target: Option<u64>,

    /// Compress distribute-stage scratch files to reduce temp-dir footprint.
    /// "none" (default) is fastest on local NVMe; "lz4" cuts disk usage
    /// ~3-4× at a small CPU cost, useful on space-constrained workers and
    /// network filesystems.
    #[arg(long, value_enum, default_value_t = TempCompressionArg::None)]
    temp_compression: TempCompressionArg,

    /// Storage layout for per-node point data during build. "files"
    /// (default) writes one temp file per octree node; simple and zero
    /// dead space but can create 100k+ files on very large inputs.
    /// "packed" writes all node data into a handful of pack files with
    /// an in-memory index — use this when the scratch filesystem has
    /// inode limits (shared storage). Trades disk space for inodes:
    /// overwrites during merge leak dead space into the packs.
    #[arg(long, value_enum, default_value_t = NodeStorageArg::Files)]
    node_storage: NodeStorageArg,
}

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum TempCompressionArg {
    #[default]
    None,
    Lz4,
}

impl From<TempCompressionArg> for TempCompression {
    fn from(a: TempCompressionArg) -> Self {
        match a {
            TempCompressionArg::None => TempCompression::None,
            TempCompressionArg::Lz4 => TempCompression::Lz4,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum NodeStorageArg {
    #[default]
    Files,
    Packed,
}

impl From<NodeStorageArg> for copc_converter::NodeStorage {
    fn from(a: NodeStorageArg) -> Self {
        match a {
            NodeStorageArg::Files => copc_converter::NodeStorage::Files,
            NodeStorageArg::Packed => copc_converter::NodeStorage::Packed,
        }
    }
}

#[derive(Debug, Clone, ValueEnum)]
enum ProgressMode {
    /// Interactive progress bar (default)
    Bar,
    /// Plain text log lines
    Plain,
    /// NDJSON — one JSON object per line
    Json,
}

/// Detect available memory from cgroup limits (v2 then v1) or system RAM.
fn detect_available_memory() -> u64 {
    // Try cgroup v2
    if let Ok(s) = std::fs::read_to_string("/sys/fs/cgroup/memory.max") {
        let s = s.trim();
        if s != "max"
            && let Ok(v) = s.parse::<u64>()
        {
            return v;
        }
    }
    // Try cgroup v1
    if let Ok(s) = std::fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        && let Ok(v) = s.trim().parse::<u64>()
    {
        // v1 returns a huge sentinel value when unlimited
        if v < 0x7FFF_FFFF_FFFF_F000 {
            return v;
        }
    }
    // Fallback: read /proc/meminfo (Linux) or use sysctl (macOS)
    #[cfg(target_os = "linux")]
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let kb_str = rest.trim().trim_end_matches(" kB").trim();
                if let Ok(kb) = kb_str.parse::<u64>() {
                    return kb * 1024;
                }
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl").arg("-n").arg("hw.memsize").output()
            && let Ok(s) = std::str::from_utf8(&output.stdout)
            && let Ok(v) = s.trim().parse::<u64>()
        {
            return v;
        }
    }
    #[cfg(target_os = "windows")]
    {
        use std::mem::MaybeUninit;
        // SAFETY: GlobalMemoryStatusEx is a well-defined Windows API call.
        unsafe {
            #[repr(C)]
            struct MemoryStatusEx {
                length: u32,
                memory_load: u32,
                total_phys: u64,
                avail_phys: u64,
                total_page_file: u64,
                avail_page_file: u64,
                total_virtual: u64,
                avail_virtual: u64,
                avail_extended_virtual: u64,
            }
            unsafe extern "system" {
                fn GlobalMemoryStatusEx(buf: *mut MemoryStatusEx) -> i32;
            }
            let mut status = MaybeUninit::<MemoryStatusEx>::zeroed().assume_init();
            status.length = std::mem::size_of::<MemoryStatusEx>() as u32;
            if GlobalMemoryStatusEx(&mut status) != 0 {
                return status.total_phys;
            }
        }
    }
    // Last resort: 16 GB
    16 * 1024 * 1024 * 1024
}

/// Parse a human-readable size string into bytes.
fn parse_memory_limit(s: &str) -> Result<u64> {
    let s = s.trim();
    let (num_part, multiplier) = if let Some(n) = s.strip_suffix(['G', 'g']) {
        (n.trim(), 1024u64 * 1024 * 1024)
    } else if let Some(n) = s.strip_suffix(['M', 'm']) {
        (n.trim(), 1024u64 * 1024)
    } else if let Some(n) = s.strip_suffix(['K', 'k']) {
        (n.trim(), 1024u64)
    } else {
        (s, 1u64)
    };
    let value: f64 = num_part
        .parse()
        .with_context(|| format!("Invalid memory limit: {s:?}"))?;
    Ok((value * multiplier as f64) as u64)
}

/// Total pipeline stages: Scanning + Counting + Distributing + Building + Writing.
const TOTAL_STEPS: u32 = 5;

fn human_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

// ---------------------------------------------------------------------------
// Bar progress (interactive terminal)
// ---------------------------------------------------------------------------

struct BarProgress {
    bar: Mutex<Option<ProgressBar>>,
    step: std::sync::atomic::AtomicU32,
    stage_prefix: Mutex<String>,
    stage_total: std::sync::atomic::AtomicU64,
    total_steps: u32,
}

impl BarProgress {
    fn new(total_steps: u32) -> Self {
        Self {
            bar: Mutex::new(None),
            step: std::sync::atomic::AtomicU32::new(0),
            stage_prefix: Mutex::new(String::new()),
            stage_total: std::sync::atomic::AtomicU64::new(0),
            total_steps,
        }
    }
}

impl ProgressObserver for BarProgress {
    fn on_progress(&self, event: ProgressEvent) {
        let mut bar = self.bar.lock().unwrap();
        let total_steps = self.total_steps;
        match event {
            ProgressEvent::StageStart { name, total } => {
                let step = self.step.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                let prefix = format!("[{step}/{total_steps}] {name}");
                *self.stage_prefix.lock().unwrap() = prefix.clone();
                self.stage_total
                    .store(total, std::sync::atomic::Ordering::Relaxed);
                let pb = if total > 0 {
                    let pb = ProgressBar::new(total);
                    pb.set_style(
                        ProgressStyle::with_template("{msg} [{bar:40}] ({eta})")
                            .unwrap()
                            .progress_chars("=> "),
                    );
                    pb.set_message(format!("{prefix} 0/{}", human_count(total)));
                    pb
                } else {
                    let pb = ProgressBar::new_spinner();
                    pb.set_style(ProgressStyle::with_template("{msg}...").unwrap());
                    pb.set_message(prefix);
                    pb
                };
                *bar = Some(pb);
            }
            ProgressEvent::StageProgress { done } => {
                if let Some(ref pb) = *bar {
                    pb.set_position(done);
                    let total = self.stage_total.load(std::sync::atomic::Ordering::Relaxed);
                    let prefix = self.stage_prefix.lock().unwrap().clone();
                    pb.set_message(format!(
                        "{prefix} {}/{}",
                        human_count(done),
                        human_count(total)
                    ));
                }
            }
            ProgressEvent::StageDone => {
                if let Some(pb) = bar.take() {
                    let prefix = self.stage_prefix.lock().unwrap().clone();
                    pb.finish_and_clear();
                    eprintln!("{prefix} done");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Plain progress (log-friendly text lines on stderr)
// ---------------------------------------------------------------------------

struct PlainProgress {
    step: std::sync::atomic::AtomicU32,
    stage_name: Mutex<String>,
    stage_total: std::sync::atomic::AtomicU64,
    last_percent: std::sync::atomic::AtomicU32,
    total_steps: u32,
}

impl PlainProgress {
    fn new(total_steps: u32) -> Self {
        Self {
            step: std::sync::atomic::AtomicU32::new(0),
            stage_name: Mutex::new(String::new()),
            stage_total: std::sync::atomic::AtomicU64::new(0),
            last_percent: std::sync::atomic::AtomicU32::new(0),
            total_steps,
        }
    }
}

/// Write a line to stdout and flush immediately so K8s log collectors see it.
fn log_line(msg: std::fmt::Arguments) {
    use std::io::Write;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(out, "{msg}");
    let _ = out.flush();
}

impl ProgressObserver for PlainProgress {
    fn on_progress(&self, event: ProgressEvent) {
        let total_steps = self.total_steps;
        match event {
            ProgressEvent::StageStart { name, total } => {
                let step = self.step.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                *self.stage_name.lock().unwrap() = name.to_string();
                self.stage_total
                    .store(total, std::sync::atomic::Ordering::Relaxed);
                self.last_percent
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                if total > 0 {
                    log_line(format_args!(
                        "[{step}/{total_steps}] {name} started ({} units)",
                        human_count(total)
                    ));
                } else {
                    log_line(format_args!("[{step}/{total_steps}] {name} started"));
                }
            }
            ProgressEvent::StageProgress { done } => {
                let total = self.stage_total.load(std::sync::atomic::Ordering::Relaxed);
                if total == 0 {
                    return;
                }
                let pct = (done as f64 / total as f64 * 100.0) as u32;
                // Log every 10%
                let bucket = pct / 10 * 10;
                let prev = self.last_percent.load(std::sync::atomic::Ordering::Relaxed);
                if bucket > prev
                    && self
                        .last_percent
                        .compare_exchange(
                            prev,
                            bucket,
                            std::sync::atomic::Ordering::Relaxed,
                            std::sync::atomic::Ordering::Relaxed,
                        )
                        .is_ok()
                {
                    let step = self.step.load(std::sync::atomic::Ordering::Relaxed);
                    let name = self.stage_name.lock().unwrap().clone();
                    log_line(format_args!(
                        "[{step}/{total_steps}] {name} {bucket}% ({}/{})",
                        human_count(done),
                        human_count(total),
                    ));
                }
            }
            ProgressEvent::StageDone => {
                let step = self.step.load(std::sync::atomic::Ordering::Relaxed);
                let name = self.stage_name.lock().unwrap().clone();
                log_line(format_args!("[{step}/{total_steps}] {name} done"));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// JSON progress (NDJSON on stdout, flushed per line)
// ---------------------------------------------------------------------------

struct JsonProgress {
    step: std::sync::atomic::AtomicU32,
    stage_name: Mutex<String>,
    stage_total: std::sync::atomic::AtomicU64,
    last_percent: std::sync::atomic::AtomicU32,
    total_steps: u32,
}

impl JsonProgress {
    fn new(total_steps: u32) -> Self {
        Self {
            step: std::sync::atomic::AtomicU32::new(0),
            stage_name: Mutex::new(String::new()),
            stage_total: std::sync::atomic::AtomicU64::new(0),
            last_percent: std::sync::atomic::AtomicU32::new(0),
            total_steps,
        }
    }

    fn emit(&self, value: &serde_json::Value) {
        log_line(format_args!("{}", serde_json::to_string(value).unwrap()));
    }
}

impl ProgressObserver for JsonProgress {
    fn on_progress(&self, event: ProgressEvent) {
        let total_steps = self.total_steps;
        match event {
            ProgressEvent::StageStart { name, total } => {
                let step = self.step.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                *self.stage_name.lock().unwrap() = name.to_string();
                self.stage_total
                    .store(total, std::sync::atomic::Ordering::Relaxed);
                self.last_percent
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                self.emit(&serde_json::json!({
                    "event": "stage_start",
                    "stage": name,
                    "step": step,
                    "total_steps": total_steps,
                    "total_units": total,
                }));
            }
            ProgressEvent::StageProgress { done } => {
                let total = self.stage_total.load(std::sync::atomic::Ordering::Relaxed);
                if total == 0 {
                    return;
                }
                let pct = (done as f64 / total as f64 * 100.0) as u32;
                let bucket = pct / 10 * 10;
                let prev = self.last_percent.load(std::sync::atomic::Ordering::Relaxed);
                if bucket > prev
                    && self
                        .last_percent
                        .compare_exchange(
                            prev,
                            bucket,
                            std::sync::atomic::Ordering::Relaxed,
                            std::sync::atomic::Ordering::Relaxed,
                        )
                        .is_ok()
                {
                    let step = self.step.load(std::sync::atomic::Ordering::Relaxed);
                    let name = self.stage_name.lock().unwrap().clone();
                    let percent = done as f64 / total as f64 * 100.0;
                    self.emit(&serde_json::json!({
                        "event": "stage_progress",
                        "stage": name,
                        "step": step,
                        "total_steps": total_steps,
                        "done": done,
                        "total": total,
                        "percent": (percent * 10.0).round() / 10.0,
                    }));
                }
            }
            ProgressEvent::StageDone => {
                let step = self.step.load(std::sync::atomic::Ordering::Relaxed);
                let name = self.stage_name.lock().unwrap().clone();
                self.emit(&serde_json::json!({
                    "event": "stage_done",
                    "stage": name,
                    "step": step,
                    "total_steps": total_steps,
                }));
            }
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_target(false)
        .init();

    let args = Args::parse();

    if let Some(n) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .context("Failed to set rayon thread count")?;
    }

    let input_files = collect_input_files(args.input)?;

    let output = if args.output.is_dir() {
        args.output.join("output.copc.laz")
    } else {
        args.output
    };

    let raw_limit = match &args.memory_limit {
        Some(s) => parse_memory_limit(s)?,
        None => detect_available_memory(),
    };
    let memory_budget = (raw_limit as f64 * MEMORY_SAFETY_FACTOR) as u64;
    let human_bytes = |b: u64| -> String {
        if b >= 1024 * 1024 * 1024 {
            format!("{:.1} GB", b as f64 / (1024.0 * 1024.0 * 1024.0))
        } else {
            format!("{:.0} MB", b as f64 / (1024.0 * 1024.0))
        }
    };
    eprintln!(
        "Memory: {} limit, {} budget ({})",
        human_bytes(raw_limit),
        human_bytes(memory_budget),
        if args.memory_limit.is_some() {
            "user-specified"
        } else {
            "auto-detected"
        },
    );

    let progress: std::sync::Arc<dyn ProgressObserver> = match args.progress {
        ProgressMode::Bar => std::sync::Arc::new(BarProgress::new(TOTAL_STEPS)),
        ProgressMode::Plain => std::sync::Arc::new(PlainProgress::new(TOTAL_STEPS)),
        ProgressMode::Json => std::sync::Arc::new(JsonProgress::new(TOTAL_STEPS)),
    };

    let config = PipelineConfig {
        memory_budget,
        temp_dir: args.temp_dir,
        temporal_index: args.temporal_index,
        progress: Some(progress),
        chunk_target_override: args.chunk_target,
        temp_compression: args.temp_compression.into(),
        node_storage: args.node_storage.into(),
    };

    let distributed = Pipeline::scan(&input_files, config)?
        .validate()?
        .distribute()?;

    if let Some(m) = distributed.header_bounds_mismatch() {
        eprintln!(
            "Warning: LAS header bounds disagree with actual point data \
             (output is still correct, but input headers are inaccurate):"
        );
        let print_axis = |name: &str, diff: Option<(f64, f64)>| {
            if let Some((h, a)) = diff {
                eprintln!("  {name}: header={h} actual={a} (diff {:+})", a - h);
            }
        };
        print_axis("min_x", m.min_x);
        print_axis("max_x", m.max_x);
        print_axis("min_y", m.min_y);
        print_axis("max_y", m.max_y);
        print_axis("min_z", m.min_z);
        print_axis("max_z", m.max_z);
    }

    distributed.build()?.write(&output)?;

    Ok(())
}
