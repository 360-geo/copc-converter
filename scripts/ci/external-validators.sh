#!/usr/bin/env bash
# Convert the test fixtures with the release binary and check every output
# with independent COPC readers:
#   - PDAL readers.copc: header summary plus a full decode of every point
#   - hobu's copc-validator, deep scan (every point of every node)
#
# Usage: scripts/ci/external-validators.sh
# Env:   BIN            converter binary (default target/release/copc_converter)
#        COPC_VALIDATOR validator command (default: npx copc-validator, pinned)
set -euo pipefail

BIN=${BIN:-target/release/copc_converter}
COPC_VALIDATOR=${COPC_VALIDATOR:-"npx --yes copc-validator@0.4.5"}
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

# copc-validator checks allowed to fail, and why:
#   wkt: copc-validator parses the SRS with proj4js, which can't read the
#        WKT2 COMPOUNDCRS in input.laz and fails outright when a file has
#        no CRS (extra_bytes_input.laz). PDAL reads both SRSs fine.
ALLOWED_FAILS='["wkt"]'

# fixture | label | converter args
CASES=(
  "input.laz|default|"
  "input.laz|chunked-merge|--memory-limit 1M"
  "input.laz|temporal-index|--temporal-index 1000"
  "extra_bytes_input.laz|default|"
)

count_points() { pdal info --summary "$1" | jq -r '.summary.num_points'; }

failed=0
for case in "${CASES[@]}"; do
  IFS='|' read -r fixture label args <<<"$case"
  name="${fixture%%.*}-${label}"
  out="$OUT/$name.copc.laz"
  echo "::group::$name"
  # shellcheck disable=SC2086 # args is a deliberately word-split flag list
  "$BIN" "tests/data/$fixture" "$out" --progress plain $args >/dev/null

  expected=$(count_points "tests/data/$fixture")
  summary=$(count_points "$out")
  # --stats forces readers.copc to decode every point of every node.
  decoded=$(pdal info --stats "$out" | jq -r '.stats.statistic[0].count')
  echo "points: input=$expected header=$summary decoded=$decoded"
  if [[ "$summary" != "$expected" || "$decoded" != "$expected" ]]; then
    echo "::error::$name: PDAL point count mismatch (input $expected, header $summary, decoded $decoded)"
    failed=1
  fi

  report="$OUT/$name.validator.json"
  $COPC_VALIDATOR --deep --output "$report" "$out" >/dev/null
  jq -r '.checks[] | "\(.status)\t\(.id)\t\(.info // "")"' "$report"
  unexpected=$(jq -r --argjson allowed "$ALLOWED_FAILS" \
    '[.checks[] | select(.status == "fail" and (.id | IN($allowed[]) | not)) | .id] | join(", ")' \
    "$report")
  if [[ -n "$unexpected" ]]; then
    echo "::error::$name: copc-validator failed: $unexpected"
    failed=1
  fi
  echo "::endgroup::"
done

exit "$failed"
