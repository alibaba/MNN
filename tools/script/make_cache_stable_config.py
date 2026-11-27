"""Create an opt-in non-thinking template without modifying the delivered model."""
import argparse,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--config',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
source=a.config.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
assert not (out/'config.runtime.json').exists(), 'Choose a new output directory'
c=json.loads(source.read_text());base=pathlib.Path(c.get('base_dir') or source.parent)
if not base.is_absolute():base=(source.parent/base).resolve()
for key in ['llm_model','llm_weight','tokenizer_file','llm_config','audio_model','audio_weight','visual_model','visual_weight']:
 if key in c and c[key]:
  value=pathlib.Path(c[key]);c[key]=str(value if value.is_absolute() else (base/value).resolve())
lc=json.loads(pathlib.Path(c['llm_config']).read_text())
old='{%- if loop.index0 > ns.last_query_index %}'
new='{%- if loop.index0 > ns.last_query_index or (enable_thinking is defined and enable_thinking is false and not reasoning_content) %}'
template=lc['jinja']['chat_template']
assert template.count(old)==1,'Not the validated Qwen3.5 template; review before adapting'
assert c.get('sampler_type')=='greedy','Validated only with greedy sampling'
lc['jinja']['chat_template']=template.replace(old,new)
(out/'llm_config.cache-stable.json').write_text(json.dumps(lc,ensure_ascii=False,indent=2))
c['llm_config']=str(out/'llm_config.cache-stable.json');c['base_dir']='';c['tmp_path']=str(out/'tmp')
c['reuse_kv']=True;c['prompt_cache']=True
c.setdefault('jinja',{}).setdefault('context',{})['enable_thinking']=False
# Pin the template in the runtime config as well as the model metadata.
c['jinja']['chat_template']=lc['jinja']['chat_template']
if 'eos' in lc['jinja']:c['jinja']['eos']=lc['jinja']['eos']
c['prompt_cache_check_template']=True
(out/'tmp').mkdir(exist_ok=True)
(out/'config.runtime.json').write_text(json.dumps(c,ensure_ascii=False,indent=2))
print(out/'config.runtime.json')
