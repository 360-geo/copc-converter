//! Preview how an input LAS/LAZ dataset would be chunked during conversion.
//!
//! Runs the hierarchical counting-sort chunk planner from
//! `copc_converter::chunking` and prints a statistics report. Does not
//! write any output — it's a read-only preview of the partitioning the
//! converter would produce for the same inputs.
//!
//! Usage:
//!   preview_chunking <input_file_or_dir> [--memory-limit 16G] [--chunk-target 5M]

use anyhow::{Context, Result};
use clap::Parser;
use copc_converter::{
    ChunkPlan, Pipeline, PipelineConfig, ProgressEvent, ProgressObserver, collect_input_files,
    select_grid_size,
};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

/// Maximum fraction of the stated memory limit to actually use.
const MEMORY_SAFETY_FACTOR: f64 = 0.75;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Analyze the chunking algorithm against a dataset",
    long_about = "Runs the hierarchical counting-sort chunk planner from the chunking \
                  module on real input files and prints chunk size distribution statistics."
)]
struct Args {
    /// Input LAZ/LAS file, or a directory containing them
    input: PathBuf,

    /// Maximum memory budget (e.g. "64G", "16G", "4096M").
    /// If not specified, auto-detects from cgroup limits or system RAM.
    #[arg(long)]
    memory_limit: Option<String>,

    /// Override the dynamically-derived chunk target size (in points).
    /// Accepts the same suffixes as memory: "10M", "5000000", "1G".
    /// Without this flag the target is computed from the budget.
    #[arg(long)]
    chunk_target: Option<String>,

    /// Print the full chunk list (one line per chunk) in addition to summary statistics.
    #[arg(long)]
    verbose: bool,

    /// Maximum number of parallel threads. Default: all available cores.
    #[arg(long)]
    threads: Option<usize>,
}

/// Detect available memory from cgroup limits (v2 then v1) or system RAM.
fn detect_available_memory() -> u64 {
    if let Ok(s) = std::fs::read_to_string("/sys/fs/cgroup/memory.max") {
        let s = s.trim();
        if s != "max"
            && let Ok(v) = s.parse::<u64>()
        {
            return v;
        }
    }
    if let Ok(s) = std::fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        && let Ok(v) = s.trim().parse::<u64>()
        && v < 0x7FFF_FFFF_FFFF_F000
    {
        return v;
    }
    #[cfg(target_os = "linux")]
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:")
                && let Ok(kb) = rest.trim().trim_end_matches(" kB").trim().parse::<u64>()
            {
                return kb * 1024;
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
    16 * 1024 * 1024 * 1024
}

/// Parse "10M", "1G", "500K", or a plain integer.
fn parse_count(s: &str) -> Result<u64> {
    let s = s.trim();
    let (num_part, multiplier) = if let Some(n) = s.strip_suffix(['G', 'g']) {
        (n.trim(), 1_000_000_000u64)
    } else if let Some(n) = s.strip_suffix(['M', 'm']) {
        (n.trim(), 1_000_000u64)
    } else if let Some(n) = s.strip_suffix(['K', 'k']) {
        (n.trim(), 1_000u64)
    } else {
        (s, 1u64)
    };
    let value: f64 = num_part
        .parse()
        .with_context(|| format!("Invalid count: {s:?}"))?;
    Ok((value * multiplier as f64) as u64)
}

/// Parse a human-readable memory size into bytes.
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

fn human_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn human_bytes(b: u64) -> String {
    if b >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", b as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if b >= 1024 * 1024 {
        format!("{:.0} MB", b as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.0} KB", b as f64 / 1024.0)
    }
}

/// Minimal stderr-only progress observer for the analyze tool.
///
/// Each stage gets a prefix line on start; intermediate progress is reported
/// at 10% increments to keep output volume sane on long runs.
struct StderrProgress {
    stage_name: Mutex<String>,
    stage_total: std::sync::atomic::AtomicU64,
    last_percent: std::sync::atomic::AtomicU32,
}

impl StderrProgress {
    fn new() -> Self {
        Self {
            stage_name: Mutex::new(String::new()),
            stage_total: std::sync::atomic::AtomicU64::new(0),
            last_percent: std::sync::atomic::AtomicU32::new(0),
        }
    }
}

impl ProgressObserver for StderrProgress {
    fn on_progress(&self, event: ProgressEvent) {
        match event {
            ProgressEvent::StageStart { name, total } => {
                *self.stage_name.lock().unwrap() = name.to_string();
                self.stage_total
                    .store(total, std::sync::atomic::Ordering::Relaxed);
                self.last_percent
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                if total > 0 {
                    eprintln!("[{name}] start ({} units)", human_count(total));
                } else {
                    eprintln!("[{name}] start");
                }
            }
            ProgressEvent::StageProgress { done } => {
                let total = self.stage_total.load(std::sync::atomic::Ordering::Relaxed);
                if total == 0 {
                    return;
                }
                let pct = (done as f64 / total as f64 * 100.0) as u32;
                let bucket = (pct / 10) * 10;
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
                    let name = self.stage_name.lock().unwrap().clone();
                    eprintln!(
                        "[{name}] {bucket}% ({}/{})",
                        human_count(done),
                        human_count(total)
                    );
                }
            }
            ProgressEvent::StageDone => {
                let name = self.stage_name.lock().unwrap().clone();
                eprintln!("[{name}] done");
            }
        }
    }
}

/// Print the chunk plan as a statistics report on stdout.
fn print_report(plan: &ChunkPlan, count_duration_secs: f64, merge_duration_secs: f64) {
    let n = plan.chunks.len();
    if n == 0 {
        println!("Chunk plan is empty (no input points).");
        return;
    }

    let mut sizes: Vec<u64> = plan.chunks.iter().map(|c| c.point_count).collect();
    sizes.sort_unstable();

    let pct = |p: f64| -> u64 {
        let idx = ((sizes.len() as f64 - 1.0) * p).round() as usize;
        sizes[idx]
    };
    let min = sizes[0];
    let max = sizes[sizes.len() - 1];
    let p50 = pct(0.50);
    let p90 = pct(0.90);
    let p99 = pct(0.99);
    let mean = sizes.iter().sum::<u64>() / n as u64;
    let variance = sizes
        .iter()
        .map(|&s| {
            let diff = s as i64 - mean as i64;
            (diff * diff) as f64
        })
        .sum::<f64>()
        / n as f64;
    let std_dev = variance.sqrt() as u64;

    let above_target = sizes.iter().filter(|&&s| s > plan.chunk_target).count();
    let above_2x = sizes.iter().filter(|&&s| s > 2 * plan.chunk_target).count();
    let below_10pct = sizes
        .iter()
        .filter(|&&s| s < plan.chunk_target / 10)
        .count();
    let zero = sizes.iter().filter(|&&s| s == 0).count();

    // Per-level breakdown
    let mut levels: std::collections::BTreeMap<u32, (usize, u64)> =
        std::collections::BTreeMap::new();
    for c in &plan.chunks {
        let entry = levels.entry(c.level).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += c.point_count;
    }

    println!();
    println!("=== Chunk plan ===");
    println!(
        "Grid:                {}³ (octree depth {})",
        plan.grid_size, plan.grid_depth
    );
    println!(
        "Chunk target size:   {} points",
        human_count(plan.chunk_target)
    );
    println!("Total input points:  {}", human_count(plan.total_points));
    println!(
        "Total chunked points: {} ({})",
        human_count(sizes.iter().sum::<u64>()),
        if sizes.iter().sum::<u64>() == plan.total_points {
            "matches input ✓"
        } else {
            "WARNING: differs from input"
        }
    );
    println!();
    println!("Chunks generated:    {}", n);
    println!();
    println!("Chunk size distribution (points):");
    println!("  Min:    {}", human_count(min));
    println!("  P50:    {}", human_count(p50));
    println!("  P90:    {}", human_count(p90));
    println!("  P99:    {}", human_count(p99));
    println!("  Max:    {}", human_count(max));
    println!("  Mean:   {}", human_count(mean));
    println!("  StdDev: {}", human_count(std_dev));
    println!();
    println!("Quality flags:");
    println!(
        "  Above target ({}):       {} ({:.2}%)",
        human_count(plan.chunk_target),
        above_target,
        above_target as f64 / n as f64 * 100.0
    );
    println!(
        "  Above 2× target ({}):    {} ({:.2}%) {}",
        human_count(2 * plan.chunk_target),
        above_2x,
        above_2x as f64 / n as f64 * 100.0,
        if above_2x > 0 {
            "← may OOM workers"
        } else {
            ""
        }
    );
    println!(
        "  Below 10% target ({}):  {} ({:.2}%) {}",
        human_count(plan.chunk_target / 10),
        below_10pct,
        below_10pct as f64 / n as f64 * 100.0,
        if below_10pct as f64 / n as f64 > 0.05 {
            "← many tiny chunks, overhead concern"
        } else {
            ""
        }
    );
    if zero > 0 {
        println!("  Zero-population chunks: {} (bug!)", zero);
    }
    println!();
    println!("Per-level breakdown:");
    println!("  level   chunks       points       avg_size");
    for (lvl, (count, total_pts)) in &levels {
        let avg = total_pts / *count as u64;
        println!(
            "  {:>5}   {:>6}    {:>10}    {:>10}",
            lvl,
            count,
            human_count(*total_pts),
            human_count(avg)
        );
    }
    println!();
    println!("Timing:");
    println!("  Count pass:  {:.1}s", count_duration_secs);
    println!("  Merge step:  {:.3}s", merge_duration_secs);
    println!(
        "  Throughput:  {} pts/s",
        human_count((plan.total_points as f64 / count_duration_secs.max(0.001)) as u64)
    );
    println!();
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
    eprintln!("Found {} input file(s)", input_files.len());

    let raw_limit = match &args.memory_limit {
        Some(s) => parse_memory_limit(s)?,
        None => detect_available_memory(),
    };
    let memory_budget = (raw_limit as f64 * MEMORY_SAFETY_FACTOR) as u64;
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

    let chunk_target_override = match &args.chunk_target {
        Some(s) => Some(parse_count(s)?),
        None => None,
    };
    if let Some(t) = chunk_target_override {
        eprintln!("Chunk target: {} points (overridden)", human_count(t));
    } else {
        eprintln!("Chunk target: dynamic (will be derived from budget)");
    }

    let progress: std::sync::Arc<dyn ProgressObserver> = std::sync::Arc::new(StderrProgress::new());

    let config = PipelineConfig {
        memory_budget,
        temp_dir: None,
        temporal_index: None,
        progress: Some(progress),
        // The analyze tool has its own --chunk-target flag; it plumbs it
        // through separately to `validated.analyze_chunking(...)` rather
        // than via the PipelineConfig field used by the convert path.
        chunk_target_override: None,
        temp_compression: copc_converter::TempCompression::None,
        node_storage: copc_converter::NodeStorage::Files,
    };

    // Run scan + validate via the standard pipeline so the analyzer sees the
    // same geometry the converter would see.
    let scanned = Pipeline::scan(&input_files, config)?;
    let validated = scanned.validate()?;

    // Time the analyze itself, broken into count + merge for the report.
    // The chunking module emits a stage progress for the count pass; we just
    // measure overall wall time and split it heuristically (the merge step is
    // O(grid_cells) and finishes in well under a second even on 512³).
    let total_start = Instant::now();
    let plan = validated.analyze_chunking(chunk_target_override)?;
    let total_elapsed = total_start.elapsed().as_secs_f64();

    // The merge step is fast and difficult to time precisely from outside
    // the module — for an early prototype it's fine to attribute almost all
    // of the elapsed time to counting and a small constant to merging.
    let pre_existing_grid_size = select_grid_size(plan.total_points, memory_budget);
    let approx_merge_secs = match pre_existing_grid_size {
        128 => 0.01,
        256 => 0.05,
        _ => 0.4,
    };
    let approx_count_secs = (total_elapsed - approx_merge_secs).max(0.0);

    print_report(&plan, approx_count_secs, approx_merge_secs);

    if args.verbose {
        println!("=== Chunk list ===");
        let mut sorted = plan.chunks.clone();
        sorted.sort_by_key(|c| (c.level, c.gx, c.gy, c.gz));
        for c in &sorted {
            println!(
                "  L{} ({:>4},{:>4},{:>4})  {} pts",
                c.level,
                c.gx,
                c.gy,
                c.gz,
                human_count(c.point_count)
            );
        }
    }

    Ok(())
}
