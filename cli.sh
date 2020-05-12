#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
NEURODECODE_PATH="$SCRIPTPATH/../NeuroDecode"

NEURODECODE_SCRIPTS="$SCRIPTPATH/neurodecode_protocols/meditation/"

if [ -z ${NEUROD_ROOT+x} ]; then
  echo "NEUROD_ROOT is not set, run '. ../NeuroDecode/env.sh' first"
  exit 1
fi

display_help() {
  echo "$(basename "$0") CLI for NFME, to be used with NeuroDecode"
  echo
  echo "-h | --help    # displays this message"
  echo
  echo
}

# for more information check this
# - https://gist.github.com/magnetikonline/22c1eb412daa350eeceee76c97519da8
# - https://gist.github.com/cosimo/3760587

opts=$(getopt o h\
  --long help,offline,online \
  --name "${0##*/}" \
  -- "$@"
)

eval set -- "$opts"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help )
      display_help
      exit
      ;;
    --online )
      echo
      echo "running online"
      echo
      cd $NEURODECODE_SCRIPTS
      PYTHONPATH="$NEURODECODE_PATH" python online.py ./sam-meditation/config_online_sam-meditation.py
      shift
      ;;
    *)
      echo "No option provided. Check help"
      display_help
      exit 1;
      ;;
  esac
done
