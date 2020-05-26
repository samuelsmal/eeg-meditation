#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
NEURODECODE_PATH="$SCRIPTPATH/../../NeuroDecode"

if [ -z ${NEUROD_ROOT+x} ]; then
  echo "NEUROD_ROOT is not set, run 'source ../..NeuroDecode/env.sh' first"
  exit 1
fi

display_help() {
  echo "$(basename "$0") CLI for NFME, to be used with NeuroDecode"
  echo
  echo "-h | --help        # displays this message"
  echo "--dvorak           # runs dvorak et al study feedback sound"
  echo "--wave-rain-pos    # runs wave and rain feedback sound"
  echo "--wave-rain-neg    # runs wave and rain feedback sound"
  echo
  echo
}

run_online() {
  PYTHONPATH="$NEURODECODE_PATH" python -m feedback_protocols.online $1
}

# for more information check this
# - https://gist.github.com/magnetikonline/22c1eb412daa350eeceee76c97519da8
# - https://gist.github.com/cosimo/3760587

opts=$(getopt \
  -o h \
  --long help,dvorak,wave-rain-pos,wave-rain-neg \
  --name "${0##*/}" \
  -- "$@"
)

eval set -- "$opts"

silent=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help )
      display_help
      exit
      ;;
    --dvorak)
      echo
      echo "running dvorak et al version"
      echo
      run_online ./protocol_configs/dvorak_study.yml
      exit 0;
      ;;
    --wave-rain-pos)
      echo
      echo "running sound snippet version"
      echo
      run_online ./protocol_configs/wave_and_rain_positive.yml
      exit 0;
      ;;
    --wave-rain-neg)
      echo
      echo "running sound snippet version"
      echo
      run_online ./protocol_configs/wave_and_rain_negative.yml
      exit 0;
      ;;
    *)
      echo "No option provided. Check help"
      display_help
      exit 1;
      ;;
  esac
done
