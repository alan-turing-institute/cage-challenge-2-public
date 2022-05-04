#!/bin/bash

log_dir="$(pwd)/logs/"
agents_dir="$(pwd)/agents/"

if [ "$#" -eq 2 ]; then
  if [ -d "$(pwd)$2" ]
  then
    log_dir="$(pwd)$2"
  else
    echo "ERROR: Directory $(pwd)$2 does not exist, please specify from root of the root of this repo"
    exit 2
  fi
else
  echo "Using logs/ as default log directory, use second argument to specify log dir"
fi

case $1 in
  baseline_sub)
	cd $agents_dir
    python3 baseline_sub_agents/train_ppo_cur.py $log_dir
    ;;

  baseline_hier)
	cd $agents_dir
    python3 baseline_sub_agents/train_hier.py $log_dir
    ;;
  decoy)
	cd $agents_dir
    python3 hier_extended/train_decoy_ppo_cur.py $log_dir
    ;;
  decoy_sub)
	cd $agents_dir
    python3 hier_extended/train_ppo_cur.py $log_dir
    ;;
  hier_decoy)
	cd $agents_dir
    python3 hier_extended/train_hier.py $log_dir
    ;;
  feudal)
	cd $agents_dir
    python3 feudal/train_feudal.py $log_dir
    ;;
  algs)
	cd $agents_dir
    python3 baseline_sub_agents/train_algos.py $log_dir
    ;;
  *)
    echo "ERROR: Invalid model name provided. Options: baseline_sub, baseline_hier, decoy, decoy_sub"
    ;;
esac