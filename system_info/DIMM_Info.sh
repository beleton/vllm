#!/usr/bin/env bash
set -euo pipefail

sudo dmidecode -t memory > ./DIMM_Name.txt

DIMM=${1:-./DIMM_Name.txt}
if (( $# >= 3 )); then
  # Keep backward compatibility with the old interface:
  #   DIMM_Info.sh DIMM_NAME OLD_MLC_PATH OUT_CSV
  OUT_CSV=$3
else
  OUT_CSV=${2:-./node_dimm_map.csv}
fi

csv_escape() {
  local s=${1//\"/\"\"}
  printf '"%s"' "$s"
}

join_by() {
  local sep=$1
  shift || true
  local first=1
  local item
  for item in "$@"; do
    if (( first )); then
      printf "%s" "$item"
      first=0
    else
      printf "%s%s" "$sep" "$item"
    fi
  done
}

declare -A channel_slots=()
declare -A populated_channels=()
declare -A sockets_seen=()

while IFS='|' read -r socket channel locator populated; do
  key="${socket}:${channel}"
  sockets_seen["$socket"]=1
  if [[ -n ${channel_slots[$key]:-} ]]; then
    channel_slots["$key"]+=$'\n'"$locator"
  else
    channel_slots["$key"]=$locator
  fi
  if [[ $populated == 1 ]]; then
    populated_channels["$key"]=1
  fi
done < <(
  awk '
  /^Memory Device$/ {
    in_dev = 1
    loc = ""
    populated = 0
    next
  }

  in_dev && /^\tLocator:/ {
    loc = $2
    next
  }

  in_dev && /^\tSize:/ {
    populated = ($2 == "No") ? 0 : 1
    next
  }

  in_dev && /^\tType Detail:/ {
    if (loc ~ /^CPU[0-9]+_/) {
      match(loc, /^CPU([0-9]+)_([^[:space:]]+)$/, m)
      if (m[0] != "") {
        slot = m[2]
        match(slot, /^C([A-Z])/, c)
        if (c[1] != "") {
          printf "%s|%s|%s|%d\n", m[1], c[1], slot, populated
        }
      }
    }
    in_dev = 0
  }
  ' "$DIMM"
)

declare -A node_socket=()
declare -A node_mem_mb=()
declare -A socket_nodes=()

while IFS=',' read -r cpu socket node; do
  [[ $cpu == \#* ]] && continue
  [[ -z $node || $node == "-" ]] && continue
  if [[ -z ${node_socket[$node]:-} ]]; then
    node_socket["$node"]=$socket
    socket_nodes["$socket"]+="${socket_nodes[$socket]:+ }$node"
  fi
done < <(lscpu -p=cpu,socket,node)

while read -r node mem; do
  node_mem_mb["$node"]=$mem
done < <(numactl --hardware | awk '/^node [0-9]+ size:/ {print $2, $4}')

channel_num_to_letter() {
  case $1 in
    0) echo "C" ;;
    1) echo "E" ;;
    2) echo "F" ;;
    3) echo "A" ;;
    4) echo "B" ;;
    5) echo "D" ;;
    6) echo "I" ;;
    7) echo "K" ;;
    8) echo "L" ;;
    9) echo "G" ;;
    10) echo "H" ;;
    11) echo "J" ;;
    *) return 1 ;;
  esac
}

declare -A active_channels=()
declare -A edac_socket_seen=()

readarray -t topology_sockets < <(printf '%s\n' "${!socket_nodes[@]}" | sort -n)
readarray -t mc_paths < <(find /sys/devices/system/edac/mc -maxdepth 1 -mindepth 1 -type d -name 'mc*' 2>/dev/null | sort -V)

if (( ${#mc_paths[@]} == ${#topology_sockets[@]} && ${#mc_paths[@]} > 0 )); then
  for i in "${!mc_paths[@]}"; do
    socket=${topology_sockets[$i]}
    mc=${mc_paths[$i]}
    for rank_dir in "$mc"/rank*; do
      [[ -d $rank_dir ]] || continue
      size=$(<"$rank_dir/size")
      (( size > 0 )) || continue
      location=$(<"$rank_dir/dimm_location")
      channel_num=$(awk '{print $4}' <<<"$location")
      letter=$(channel_num_to_letter "$channel_num") || continue
      active_channels["${socket}:${letter}"]=1
      edac_socket_seen["$socket"]=1
    done
  done
fi

CHANNEL_ORDER=(A B C D E F G H I J K L)
QUADRANT_0=(C E F)
QUADRANT_1=(A B D)
QUADRANT_2=(I K L)
QUADRANT_3=(G H J)

channel_in_list() {
  local target=$1
  shift
  local ch
  for ch in "$@"; do
    if [[ $ch == "$target" ]]; then
      return 0
    fi
  done
  return 1
}

sorted_channel_slots() {
  local socket=$1
  local channel=$2
  local key="${socket}:${channel}"
  if [[ -n ${channel_slots[$key]:-} ]]; then
    printf '%s\n' "${channel_slots[$key]}" | sort
    return 0
  fi
  printf 'C%sD0\n' "$channel"
}

quadrant_channel_letters() {
  case $1 in
    0) printf '%s\n' "${QUADRANT_0[@]}" ;;
    1) printf '%s\n' "${QUADRANT_1[@]}" ;;
    2) printf '%s\n' "${QUADRANT_2[@]}" ;;
    3) printf '%s\n' "${QUADRANT_3[@]}" ;;
    *) return 1 ;;
  esac
}

channel_is_populated() {
  local socket=$1
  local channel=$2
  local key="${socket}:${channel}"

  if [[ -n ${populated_channels[$key]:-} ]]; then
    return 0
  fi

  if [[ -n ${active_channels[$key]:-} && -z ${channel_slots[$key]:-} ]]; then
    return 0
  fi

  return 1
}

group_slots_csv() {
  local socket=$1
  shift
  local channels=()
  local q
  local q_channels=()
  local channel
  local slots=()
  local slot

  for q in "$@"; do
    readarray -t q_channels < <(quadrant_channel_letters "$q")
    channels+=("${q_channels[@]}")
  done

  for channel in "${CHANNEL_ORDER[@]}"; do
    if ! channel_in_list "$channel" "${channels[@]}"; then
      continue
    fi
    if ! channel_is_populated "$socket" "$channel"; then
      continue
    fi
    while IFS= read -r slot; do
      [[ -z $slot ]] && continue
      slots+=("$slot")
    done < <(sorted_channel_slots "$socket" "$channel")
  done

  join_by "," "${slots[@]}"
}

group_channel_count() {
  local socket=$1
  shift
  local channels=()
  local q
  local q_channels=()
  local channel
  local count=0

  for q in "$@"; do
    readarray -t q_channels < <(quadrant_channel_letters "$q")
    channels+=("${q_channels[@]}")
  done

  for channel in "${CHANNEL_ORDER[@]}"; do
    if ! channel_in_list "$channel" "${channels[@]}"; then
      continue
    fi
    if channel_is_populated "$socket" "$channel"; then
      ((count += 1))
    fi
  done

  echo "$count"
}

warn_mismatched_inventory() {
  local socket=$1
  local dmi_count=0
  local edac_count=0
  local channel

  [[ -n ${edac_socket_seen[$socket]:-} ]] || return 0

  for channel in "${CHANNEL_ORDER[@]}"; do
    [[ -n ${populated_channels["${socket}:${channel}"]:-} ]] && ((dmi_count += 1))
    [[ -n ${active_channels["${socket}:${channel}"]:-} ]] && ((edac_count += 1))
  done

  if (( dmi_count != edac_count )); then
    echo "Warning: socket ${socket} inventory mismatch, DIMM file shows ${dmi_count} populated channels but EDAC shows ${edac_count}; using DIMM inventory for slot mapping." >&2
  fi
}

unsupported_layout() {
  local socket=$1
  local nodes_per_socket=$2
  echo "Unsupported NUMA layout on socket ${socket}: ${nodes_per_socket} nodes per socket. Only NPS1/NPS2/NPS4 are supported." >&2
  exit 1
}

{
  echo "node,socket,node_mem_mb,channels_infer,dimms_infer"

  readarray -t all_sockets < <(printf '%s\n' "${!sockets_seen[@]}" | sort -n)
  for socket in "${all_sockets[@]}"; do
    if [[ -z ${socket_nodes[$socket]:-} ]]; then
      continue
    fi

    warn_mismatched_inventory "$socket"

    readarray -t nodes < <(printf '%s\n' ${socket_nodes[$socket]} | sort -n)
    nodes_per_socket=${#nodes[@]}

    case $nodes_per_socket in
      1)
        groups=("0 1 2 3")
        ;;
      2)
        # AMD EPYC 9005 uses 4x3-UMC quadrants. In NPS2, each NUMA node spans
        # two adjacent quadrants on one side of the socket: A-F and G-L.
        groups=("0 1" "2 3")
        ;;
      4)
        # Quadrants derived from AMD's UMC/channel topology:
        #   q0 = A/B/D, q1 = C/E/F, q2 = I/K/L, q3 = G/H/J
        groups=("0" "1" "2" "3")
        ;;
      *)
        unsupported_layout "$socket" "$nodes_per_socket"
        ;;
    esac

    for i in "${!nodes[@]}"; do
      node=${nodes[$i]}
      group_spec=${groups[$i]}
      # shellcheck disable=SC2206
      quadrants=($group_spec)
      mem=${node_mem_mb[$node]:-}
      dimms=$(group_slots_csv "$socket" "${quadrants[@]}")
      ch=$(group_channel_count "$socket" "${quadrants[@]}")

      printf "%s,%s,%s,%s,%s\n" \
        "node${node}" \
        "CPU${socket}" \
        "$mem" \
        "$ch" \
        "$(csv_escape "$dimms")"
    done
  done
} > "$OUT_CSV"

echo "CSV written to: $OUT_CSV"
