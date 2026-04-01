ps -u "$USER" -eo pid=,args= | awk 'BEGIN{IGNORECASE=1} /vllm|worker_tp/ {print $1}' | xargs -r kill -9

find /dev/shm -maxdepth 1 -regextype posix-extended \
    -regex '.*/[0-9]+-tp:0-\[0-1\]-cpushm_[01]' -delete