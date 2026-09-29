#!/usr/bin/env bash
# Check the converter stays inside its memory budget by running it in
# containers with a hard cgroup memory limit, the way a Kubernetes pod runs
# it. An OOM kill fails the case, just as it would OOMKill the pod. Without
# --memory-limit the converter must also detect the limit from the cgroup.
#
# Usage: scripts/ci/memory-limits.sh [gating|known-issues]
#   gating        cases that must stay within the limit; any OOM fails (default)
#   known-issues  inputs known to exceed the limit today; reported, never
#                 fails. Move a case to gating once its fix lands.
# Env:   BIN   Linux converter binary (default target/release/copc_converter)
#        CPUS  container CPU limit (default 4, a GitHub runner's core count);
#              memory use scales with the thread count derived from it
#        WORK  scratch dir for inputs and outputs (default: fresh temp dir);
#              reuse it across runs to skip regenerating inputs
set -euo pipefail

MODE=${1:-gating}
CPUS=${CPUS:-4}
BIN=$(realpath "${BIN:-target/release/copc_converter}")
WORK=${WORK:-$(mktemp -d)}
mkdir -p "$WORK"
WORK=$(realpath "$WORK")
IMAGE=copc-converter-memtest

# GNU time inside the container records the converter's peak RSS.
docker build --quiet --tag "$IMAGE" - >/dev/null <<'EOF'
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends time \
 && rm -rf /var/lib/apt/lists/*
EOF

mkdir -p "$WORK/in" "$WORK/out"

# faux NAME COUNT MODE BOUNDS [EXTRA_DIM_COUNT]
# Write a synthetic LAZ input with PDAL's readers.faux (skipped if present).
# EXTRA_DIM_COUNT adds that many f64 Extra Bytes dimensions per point.
faux() {
  local name=$1 count=$2 mode=$3 bounds=$4 extra=${5:-0}
  local path="$WORK/in/$name.laz"
  [[ -f "$path" ]] && return
  local ferry="" assign='"GpsTime = X*0.001 + Y","ReturnNumber = 1","NumberOfReturns = 1"'
  if ((extra > 0)); then
    local dims=()
    for ((i = 0; i < extra; i++)); do
      dims+=("\"=>E$i\"")
      assign+=",\"E$i = X*$((i + 1)) + Y\""
    done
    ferry=$(IFS=,; echo "{\"type\":\"filters.ferry\",\"dimensions\":[${dims[*]}]},")
  fi
  mkdir -p "$(dirname "$path")"
  # PDAL warns once per axis that "auto" offsets are picked in stream mode;
  # that's intended here, so keep those lines out of the job log.
  pdal pipeline --stdin 2> >(grep -v "Auto offset" >&2) <<EOF
{"pipeline":[
 {"type":"readers.faux","mode":"$mode","count":$count,"bounds":"$bounds"},
 $ferry
 {"type":"filters.assign","value":[$assign]},
 {"type":"writers.las","filename":"$path","compression":true,"extra_dims":"all",
  "minor_version":4,"dataformat_id":6,"scale_x":0.001,"scale_y":0.001,"scale_z":0.001,
  "offset_x":"auto","offset_y":"auto","offset_z":"auto"}
]}
EOF
}

# tiles NAME COUNT_PER_TILE GRID: a GRID x GRID set of adjacent 100 m tiles.
tiles() {
  local name=$1 count=$2 grid=$3
  for ((tx = 0; tx < grid; tx++)); do
    for ((ty = 0; ty < grid; ty++)); do
      local x=$((100000 + tx * 100)) y=$((400000 + ty * 100))
      faux "$name/t_${tx}_${ty}" "$count" random "([$x,$((x + 100))],[$y,$((y + 100))],[0,30])"
    done
  done
}

failed=0
summary="| case | container limit | converter args | peak RSS | result |
|---|---|---|---|---|"

# run_case NAME CONTAINER_MEMORY INPUT [converter args...]
run_case() {
  local name=$1 mem=$2 input=$3
  shift 3
  local container="copc-memtest-$name-$$" status oom rss_kb rss verdict
  set +e
  # Run as the invoking user: files the converter leaves behind (e.g. temp
  # files after an OOM kill) must be removable by this script on Linux,
  # where a root container would own them.
  docker run --name "$container" --user "$(id -u):$(id -g)" --memory="$mem" --memory-swap="$mem" --cpus="$CPUS" \
    -v "$BIN:/usr/local/bin/copc_converter:ro" -v "$WORK:/work" "$IMAGE" \
    /usr/bin/time -f '%M' -o "/work/out/$name.rss" \
    copc_converter "/work/in/$input" "/work/out/$name.copc.laz" \
    --temp-dir "/work/tmp-$name" --progress plain "$@" >"$WORK/out/$name.log" 2>&1
  status=$?
  set -e
  oom=$(docker inspect --format '{{.State.OOMKilled}}' "$container")
  docker rm "$container" >/dev/null
  rss_kb=$(grep -E '^[0-9]+$' "$WORK/out/$name.rss" 2>/dev/null | tail -1 || true)
  rss=${rss_kb:+$((rss_kb / 1024)) MB}
  rm -rf "$WORK/out/$name.copc.laz" "$WORK/tmp-$name"

  if [[ "$oom" == "true" || $status -eq 137 ]]; then
    verdict="OOM-killed"
  elif [[ $status -ne 0 ]]; then
    verdict="failed (exit $status)"
  else
    verdict="ok"
  fi
  printf '%-24s %-6s %-22s %-10s %s\n' "$name" "$mem" "${*:-(auto-detect)}" "${rss:-n/a}" "$verdict"
  summary+=$'\n'"| $name | $mem | ${*:-(auto-detect)} | ${rss:-n/a} | $verdict |"

  if [[ "$MODE" == gating && "$verdict" != ok ]]; then
    echo "::error::$name: $verdict under a $mem container limit"
    tail -20 "$WORK/out/$name.log"
    failed=1
  elif [[ "$MODE" == known-issues && "$verdict" == ok ]]; then
    echo "::notice::$name now stays within $mem; move it to the gating cases"
  fi
}

case "$MODE" in
  gating)
    faux uniform-20m 20000000 random "([0,2000],[0,2000],[0,50])"
    tiles tiles-10m 250000 6
    run_case uniform-20m-auto 1g uniform-20m.laz
    run_case uniform-20m-explicit 1g uniform-20m.laz --memory-limit 1G
    run_case tiles-10m-auto 1g tiles-10m
    ;;
  known-issues)
    # All three exceed 1 GB today (memory audit): coincident points pile
    # into one node past the depth cap; a volumetric cube's merge parents
    # exceed the budget; the writer window ignores Extra Bytes size.
    faux coincident-20m 20000000 constant "([100,100],[200,200],[5,5])"
    faux cube-20m 20000000 random "([0,1000],[0,1000],[0,1000])"
    faux extra-bytes-10m 10000000 random "([0,2000],[0,2000],[0,50])" 24
    run_case coincident-20m 1g coincident-20m.laz
    run_case cube-20m 1g cube-20m.laz
    run_case extra-bytes-10m 1g extra-bytes-10m.laz
    ;;
  *)
    echo "unknown mode: $MODE (expected gating or known-issues)" >&2
    exit 2
    ;;
esac

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  printf '### Memory limits: %s\n\n%s\n' "$MODE" "$summary" >>"$GITHUB_STEP_SUMMARY"
fi
exit "$failed"
