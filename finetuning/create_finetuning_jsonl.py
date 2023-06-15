import json

# {"prompt":"Q: What is the symbol often used to represent the word 'growth'? Give me one word.\nA:","completion":" seedling\n"}
finetune_text = "Q: What is the symbol often used to represent the word '{}'? Give me one word.\nA:"

with open('temp.json', mode='r') as f:
  a = json.loads(f.read())
b = dict()
for pair in a:
  rep = b.get(pair[0].lower(), [])
  exists = False
  pairl = pair[1].lower()
  for conw in rep:
    if pairl[:-1].replace(' ', '') in conw.replace(' ', ''):
      rep.remove(conw)
      rep.append(pairl)
      break
    elif conw[:-1].replace(' ', '') in pairl.replace(' ', ''):
      break
  else:
    rep.append(pairl)
  b[pair[0].lower()] = rep

ot = list()
with open('finetunemodel.jsonl', mode='w+') as g:
  for k,vl in b.items():
    for v in vl:
      ot.append([k, v])
  c = [{"prompt": finetune_text.format(i[0]), "completion": " {}\n".format(i[1])} for i in ot]
  output = json.dumps(c, ensure_ascii=False).replace('}, ', '}\n')[1:-1]
  g.write(output)
