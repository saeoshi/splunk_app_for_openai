# $SPLUNK_HOME/etc/apps/<your_app>/bin/openai_insight.py
from splunklib.searchcommands import dispatch, StreamingCommand, Configuration, Option, validators
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import re
import time
from collections import defaultdict

@Configuration()
class OpenAIInsightCommand(StreamingCommand):
    fields = Option(name="fields", require=True)
    mode   = Option(name="mode", require=False, default="summarize", validate=validators.Set("summarize","root_cause","qa","classify"))
    group_by = Option(name="group_by", require=False)
    batch_size = Option(name="batch_size", require=False, default=50, validate=validators.Integer(minimum=1, maximum=500))
    window_sec = Option(name="window_sec", require=False, validate=validators.Integer(minimum=0))
    redact = Option(name="redact", require=False, default="(?i)(api[_-]?key|password|token)=\\S+")

    def stream(self, records):
        # .envをロード
        load_dotenv()

        # APIキーとモデル名を.envから取得
        api_key = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("OPENAI_MODEL")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")
        if not model_name:
            raise ValueError("OPENAI_MODEL is not set. Please check your .env file.")

        client = OpenAI(api_key=api_key)

        field_list = [f.strip() for f in self.fields.split(",") if f.strip()]
        group_field = self.group_by.strip() if self.group_by else None
        redact_re = re.compile(self.redact) if self.redact else None

        buffers = defaultdict(list)
        first_ts = {}

        def flush(key):
            batch = buffers[key]
            if not batch:
                return []
            # データをJSON化
            items = []
            for r in batch:
                item = {f: r.get(f, "") for f in field_list}
                if redact_re:
                    for k, v in item.items():
                        if isinstance(v, str):
                            item[k] = redact_re.sub(r"\1=***", v)
                items.append(item)

            # OpenAIへのリクエスト
            system_prompt = (
                "You are a log analyst. Return STRICT JSON with keys: "
                "insights[], anomalies[], actions[]."
            )
            user_prompt = json.dumps({
                "mode": self.mode,
                "sample_count": len(items),
                "data": items[:self.batch_size]
            }, ensure_ascii=False)

            resp = client.chat.completions.create(
                model=model_name,  # ← .envから取得したモデル名を使用
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            )

            out = resp.choices[0].message.content
            try:
                parsed = json.loads(out)
            except Exception:
                parsed = {"insights":[{"title":"ParseError","detail":out,"confidence":0.3}]}

            result = {
                "_group": key,
                "_count": len(items),
                "insights": json.dumps(parsed.get("insights",[]), ensure_ascii=False),
                "anomalies": json.dumps(parsed.get("anomalies",[]), ensure_ascii=False),
                "actions": json.dumps(parsed.get("actions",[]), ensure_ascii=False),
            }
            buffers[key] = []
            first_ts.pop(key, None)
            return [result]

        for r in records:
            key = r.get(group_field, "__all__") if group_field else "__all__"
            buffers[key].append(r)
            if key not in first_ts:
                first_ts[key] = time.time()
            if len(buffers[key]) >= int(self.batch_size) or (self.window_sec and (time.time()-first_ts[key] >= int(self.window_sec))):
                for out in flush(key):
                    yield out

        for key in list(buffers.keys()):
            for out in flush(key):
                yield out

dispatch(OpenAIInsightCommand, module_name=__name__)
