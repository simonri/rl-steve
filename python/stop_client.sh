#!/bin/sh
# Stops every java process for one named client, e.g. stop_client.sh bot1. The lockstep can block a
# normal shutdown, so it escalates to SIGKILL after 5 seconds. Runs hold no world data worth saving.
pids=""
for pid in $(pgrep -x java); do
  if tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "runs/$1 "; then pids="$pids $pid"; fi
done
[ -z "$pids" ] && exit 0
kill $pids 2>/dev/null
i=0
while [ $i -lt 5 ]; do
  sleep 1
  alive=""
  for pid in $pids; do [ -d "/proc/$pid" ] && alive="$alive $pid"; done
  [ -z "$alive" ] && break
  i=$((i + 1))
done
[ -n "$alive" ] && kill -9 $alive 2>/dev/null
sleep 1
echo "stopped $1:$pids"
