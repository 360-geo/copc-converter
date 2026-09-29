//! Finding the memory limit the converter has to stay within.
//!
//! Shared by the `copc_converter` and `preview_chunking` binaries (not part
//! of the library API). In a Kubernetes pod the binding limit is the
//! container's cgroup, which is not necessarily the cgroup mounted at
//! `/sys/fs/cgroup`: without a private cgroup namespace that is the host's
//! root, and reading only it would fall back to the node's whole RAM.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

/// A detected memory limit and where it came from, for the log line.
pub struct DetectedLimit {
    pub bytes: u64,
    pub source: String,
}

/// Detect the memory limit: the tightest cgroup limit on this process
/// (cgroup v2 `memory.max`/`memory.high`, or v1 `memory.limit_in_bytes`, on
/// its own cgroup and every ancestor), capped at physical RAM.
pub fn detect() -> DetectedLimit {
    let cgroup = cgroup_limit(Path::new("/proc/self/cgroup"), Path::new("/sys/fs/cgroup"));
    let ram = physical_memory();
    match (cgroup, ram) {
        (Some((limit, source)), Some(ram)) if limit < ram => DetectedLimit {
            bytes: limit,
            source,
        },
        (Some((limit, source)), None) => DetectedLimit {
            bytes: limit,
            source,
        },
        (_, Some(ram)) => DetectedLimit {
            bytes: ram,
            source: "system RAM".into(),
        },
        (None, None) => DetectedLimit {
            bytes: 16 * 1024 * 1024 * 1024,
            source: "default; no limit detected".into(),
        },
    }
}

/// Values at or above this in cgroup v1's `memory.limit_in_bytes` mean
/// "unlimited" (the kernel reports a page-aligned `i64::MAX`).
const CGROUP_V1_UNLIMITED: u64 = 0x7FFF_FFFF_FFFF_F000;

/// The tightest memory limit on the process's cgroup and its ancestors, with
/// a description of where it was found. `proc_cgroup` is the contents'
/// source (`/proc/self/cgroup`) and `root` the cgroup mount (`/sys/fs/cgroup`).
///
/// Every ancestor's limit applies to the process, so the minimum along the
/// path is the one that counts. When the listed path isn't visible under
/// `root` (e.g. a container with a private cgroup namespace, where the file
/// still shows the host path), the walk starts at `root` itself.
fn cgroup_limit(proc_cgroup: &Path, root: &Path) -> Option<(u64, String)> {
    let listing = std::fs::read_to_string(proc_cgroup).ok()?;
    let mut best: Option<(u64, String)> = None;
    let mut consider = |value: u64, file: &Path| {
        if best.as_ref().is_none_or(|(b, _)| value < *b) {
            best = Some((value, format!("cgroup {}", file.display())));
        }
    };
    for line in listing.lines() {
        // "hierarchy-id:controllers:path"; v2 is "0::path".
        let mut parts = line.splitn(3, ':');
        let (Some(id), Some(controllers), Some(path)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        if id == "0" && controllers.is_empty() {
            for dir in ancestors(root, path) {
                for name in ["memory.max", "memory.high"] {
                    let file = dir.join(name);
                    if let Some(v) = read_limit(&file).filter(|v| *v != u64::MAX) {
                        consider(v, &file);
                    }
                }
            }
        } else if controllers.split(',').any(|c| c == "memory") {
            for dir in ancestors(&root.join("memory"), path) {
                let file = dir.join("memory.limit_in_bytes");
                if let Some(v) = read_limit(&file).filter(|v| *v < CGROUP_V1_UNLIMITED) {
                    consider(v, &file);
                }
            }
        }
    }
    best
}

/// `base/path` and each of its parents down to `base`; just `base` when
/// `base/path` doesn't exist.
fn ancestors(base: &Path, path: &str) -> Vec<PathBuf> {
    let leaf = base.join(path.trim_start_matches('/'));
    if !leaf.is_dir() {
        return vec![base.to_path_buf()];
    }
    leaf.ancestors()
        .take_while(|d| d.starts_with(base))
        .map(Path::to_path_buf)
        .collect()
}

/// Read a cgroup limit file: a byte count, or `max` for no limit (returned
/// as `u64::MAX`). `None` when the file is missing or unreadable.
fn read_limit(file: &Path) -> Option<u64> {
    let s = std::fs::read_to_string(file).ok()?;
    let s = s.trim();
    if s == "max" {
        Some(u64::MAX)
    } else {
        s.parse().ok()
    }
}

/// Total physical memory, if the platform reports it.
fn physical_memory() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kb = meminfo
            .lines()
            .find_map(|l| l.strip_prefix("MemTotal:"))?
            .trim()
            .trim_end_matches("kB")
            .trim()
            .parse::<u64>()
            .ok()?;
        Some(kb * 1024)
    }
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        std::str::from_utf8(&output.stdout)
            .ok()?
            .trim()
            .parse()
            .ok()
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
            (GlobalMemoryStatusEx(&mut status) != 0).then_some(status.total_phys)
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

/// Parse a size such as `16G`, `16Gi`, `16GiB`, `4096M`, `1.5T` or a plain
/// byte count. Unit prefixes are binary (`G` = GiB), with or without `i`/`B`,
/// so a Kubernetes quantity like `16Gi` can be passed straight through.
pub fn parse_size(s: &str) -> Result<u64> {
    let s = s.trim();
    let split = s.find(|c: char| c.is_ascii_alphabetic()).unwrap_or(s.len());
    let (number, unit) = s.split_at(split);
    let multiplier: u64 = match unit.to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "k" | "ki" | "kb" | "kib" => 1 << 10,
        "m" | "mi" | "mb" | "mib" => 1 << 20,
        "g" | "gi" | "gb" | "gib" => 1 << 30,
        "t" | "ti" | "tb" | "tib" => 1 << 40,
        _ => bail!("Invalid memory limit {s:?}: unknown unit {unit:?}"),
    };
    let value: f64 = number
        .trim()
        .parse()
        .with_context(|| format!("Invalid memory limit: {s:?}"))?;
    if !(value.is_finite() && value > 0.0) {
        bail!("Invalid memory limit {s:?}: must be a positive size");
    }
    Ok((value * multiplier as f64) as u64)
}

/// Whether `dir` is on a RAM-backed filesystem (tmpfs/ramfs), where scratch
/// files consume memory — and, in a container, count against its limit.
#[cfg(target_os = "linux")]
pub fn is_ram_backed(dir: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    const TMPFS_MAGIC: i64 = 0x0102_1994;
    const RAMFS_MAGIC: i64 = 0x8584_58f6;
    let Ok(path) = std::ffi::CString::new(dir.as_os_str().as_bytes()) else {
        return false;
    };
    let mut stat = std::mem::MaybeUninit::<libc::statfs>::zeroed();
    // SAFETY: `path` is a valid NUL-terminated string and `stat` is a
    // correctly sized out-parameter.
    if unsafe { libc::statfs(path.as_ptr(), stat.as_mut_ptr()) } != 0 {
        return false;
    }
    // SAFETY: statfs succeeded, so it initialised `stat`.
    // `f_type`'s integer type differs between targets (i64 on x86_64, i32 on
    // some 32-bit ones), so widen it explicitly before comparing.
    #[allow(clippy::unnecessary_cast)]
    let f_type = unsafe { stat.assume_init() }.f_type as i64;
    f_type == TMPFS_MAGIC || f_type == RAMFS_MAGIC
}

#[cfg(not(target_os = "linux"))]
pub fn is_ram_backed(_dir: &Path) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn parse_size_accepts_cli_and_kubernetes_forms() {
        for (input, expected) in [
            ("16G", 16 * GIB),
            ("16g", 16 * GIB),
            ("16Gi", 16 * GIB),
            ("16GiB", 16 * GIB),
            ("16 GB", 16 * GIB),
            ("512Mi", 512 << 20),
            ("4096M", 4 * GIB),
            ("1.5T", (1.5 * (1u64 << 40) as f64) as u64),
            ("2048", 2048),
        ] {
            assert_eq!(parse_size(input).unwrap(), expected, "{input}");
        }
        for bad in ["", "G", "16X", "-1G", "0", "abc"] {
            assert!(parse_size(bad).is_err(), "{bad:?} should be rejected");
        }
    }

    /// A fake `/proc/self/cgroup` and cgroup mount under a temp dir.
    struct FakeCgroup {
        dir: PathBuf,
    }

    impl FakeCgroup {
        fn new(name: &str, proc_cgroup: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("copc_memlimit_{name}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(dir.join("root")).unwrap();
            std::fs::write(dir.join("proc_cgroup"), proc_cgroup).unwrap();
            Self { dir }
        }

        fn set(&self, rel: &str, file: &str, value: &str) {
            let d = self.dir.join("root").join(rel);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join(file), value).unwrap();
        }

        fn limit(&self) -> Option<u64> {
            cgroup_limit(&self.dir.join("proc_cgroup"), &self.dir.join("root")).map(|(v, _)| v)
        }
    }

    impl Drop for FakeCgroup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn v2_uses_own_cgroup_not_just_the_mount_root() {
        // No private cgroup namespace: the container's limit sits in a
        // nested cgroup, and the mount root has none.
        let cg = FakeCgroup::new("v2_nested", "0::/kubepods/pod1/ctr\n");
        cg.set("kubepods/pod1/ctr", "memory.max", "2147483648\n");
        cg.set("kubepods/pod1", "memory.max", "max\n");
        assert_eq!(cg.limit(), Some(2 * GIB));
    }

    #[test]
    fn v2_takes_tightest_of_max_high_and_ancestors() {
        let cg = FakeCgroup::new("v2_min", "0::/kubepods/pod1/ctr\n");
        cg.set("kubepods/pod1/ctr", "memory.max", "max\n");
        cg.set("kubepods/pod1/ctr", "memory.high", "3221225472\n");
        cg.set("kubepods/pod1", "memory.max", "4294967296\n");
        assert_eq!(cg.limit(), Some(3 * GIB));
        cg.set("kubepods", "memory.max", "1073741824\n");
        assert_eq!(cg.limit(), Some(GIB));
    }

    #[test]
    fn v2_namespaced_container_reads_mount_root() {
        // Private cgroup namespace: the container sees itself as "/".
        let cg = FakeCgroup::new("v2_ns", "0::/\n");
        cg.set("", "memory.max", "1073741824\n");
        assert_eq!(cg.limit(), Some(GIB));
    }

    #[test]
    fn v2_unlimited_is_none() {
        let cg = FakeCgroup::new("v2_none", "0::/user.slice\n");
        cg.set("user.slice", "memory.max", "max\n");
        cg.set("user.slice", "memory.high", "max\n");
        assert_eq!(cg.limit(), None);
    }

    #[test]
    fn v1_memory_controller_and_unlimited_sentinel() {
        let cg = FakeCgroup::new(
            "v1",
            "12:cpu,cpuacct:/kubepods/pod1/ctr\n11:memory:/kubepods/pod1/ctr\n",
        );
        cg.set(
            "memory/kubepods/pod1/ctr",
            "memory.limit_in_bytes",
            "536870912\n",
        );
        cg.set("memory", "memory.limit_in_bytes", "9223372036854771712\n");
        assert_eq!(cg.limit(), Some(512 << 20));
    }

    #[test]
    fn listed_path_missing_falls_back_to_mount_root() {
        // v1 container without a cgroup namespace view of its own path.
        let cg = FakeCgroup::new("v1_fallback", "11:memory:/docker/abc\n");
        cg.set("memory", "memory.limit_in_bytes", "1073741824\n");
        assert_eq!(cg.limit(), Some(GIB));
    }

    #[test]
    fn detect_is_capped_by_physical_memory() {
        let detected = detect();
        if let Some(ram) = physical_memory() {
            assert!(detected.bytes <= ram);
        }
        assert!(detected.bytes > 0);
    }
}
