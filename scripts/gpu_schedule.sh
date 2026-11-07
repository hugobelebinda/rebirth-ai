#!/usr/bin/env bash
# GPU working hours (user's schedule, 2026-11-05):
#   start 06:00 every day
#   stop  23:00 on Sun–Thu nights; 01:00 (next day) on Fri and Sat nights
# Stopped/started by hand (no timers): see scripts/gpu_job.sh.
#
# Sourced by GPU jobs: `minutes_until_stop` prints minutes until the next scheduled stop
# (0 outside working hours).
minutes_until_stop() {
  local now dow hm stop
  now=$(date +%s); dow=$(date +%u); hm=$((10#$(date +%H) * 60 + 10#$(date +%M)))
  if [ "$hm" -lt 360 ]; then
    # 00:00–06:00: only the tail of a Fri/Sat late night (today is Sat/Sun) is working time.
    if { [ "$dow" -eq 6 ] || [ "$dow" -eq 7 ]; } && [ "$hm" -lt 60 ]; then echo $((60 - hm)); else echo 0; fi
    return
  fi
  if [ "$dow" -eq 5 ] || [ "$dow" -eq 6 ]; then stop=$((25 * 60)); else stop=$((23 * 60)); fi
  echo $(( stop > hm ? stop - hm : 0 ))
}
