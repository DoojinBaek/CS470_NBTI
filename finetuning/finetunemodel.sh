source ../.env
if [[ -z "${OPENAI_API_KEY}" ]]; then
  echo no OPENAI_API_KEY env defined
else
  openai --api-key $OPENAI_API_KEY api fine_tunes.create -t finetuning.jsonl -m davinci2
fi