  BASE_URL=http://127.0.0.1:8122
  MODEL=Qwen3-30B-A3B
  TOKENIZER=/models/Qwen3-30B-A3B
  OUTDIR=/home/zjj/vllm/test_results/Qwen3-30B-A3B/NPS1_TP2/bench_res

  parallels=(2 8 16 64)
  numbers=(2 8 16 64)

  for i in "${!parallels[@]}"; do
    c="${parallels[$i]}"
    n="${numbers[$i]}"

    vllm bench serve \
      --base-url "$BASE_URL" \
      --endpoint /v1/chat/completions \
      --backend openai-chat \
      --model "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --dataset-name random \
      --random-input-len 1024 \
      --random-output-len 1024 \
      --random-prefix-len 0 \
      --max-concurrency "$c" \
      --num-prompts "$n" \
      --request-rate inf \
      --ignore-eos \
      --save-result \
      --result-dir "$OUTDIR" \
      --result-filename "c${c}_n${n}.json"
  done