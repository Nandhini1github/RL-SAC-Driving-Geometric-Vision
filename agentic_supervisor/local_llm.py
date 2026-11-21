"""Run a bounded local LLM curriculum arm against the existing seed-37 fork."""
import json
from urllib.request import Request, urlopen

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from driving_environment.route import RouteEnv, SCENES
from sac_training.experiment import OUT, SEED, STAGE, evaluate, save_csv
from agentic_supervisor.curriculum import propose


def decide(rows):
    metrics={}
    for scene in SCENES:
        cases=[r for r in rows if r["scene"]==scene]
        metrics[scene]={k:round(float(np.mean([r[k] for r in cases])),4)
                        for k in ("success","collision","departure","rmse_m","steering_delta")}
    schema={"type":"object","properties":{"target":{"type":"string","enum":list(SCENES)},
            "reason":{"type":"string"}},"required":["target","reason"]}
    prompt=("You supervise SAC on a short route. Select one scenario to practice more, using only these "
            "camera-derived validation means. Collision is a visual near-hazard proxy, and success is visual odometry progress. Prioritize hazards and failure to finish, then lane tracking and steering. "
            "Give one short reason grounded in the numbers.\n"+json.dumps(metrics))
    payload=dict(model="qwen2.5:7b", prompt=prompt, format=schema, stream=False,
                 options=dict(temperature=0,seed=SEED,num_predict=150,num_ctx=2048))
    request=Request("http://127.0.0.1:11434/api/generate",data=json.dumps(payload).encode(),
                    headers={"Content-Type":"application/json"})
    try:
        with urlopen(request,timeout=150) as response:
            answer=json.load(response)
        choice=json.loads(answer["response"])
        target=choice["target"]
        if target not in SCENES or not isinstance(choice["reason"],str):
            raise ValueError("Invalid bounded choice")
        return dict(source=payload["model"],target=target,reason=choice["reason"],metrics=metrics,
                    weights=[0.6 if x==target else 0.2 for x in SCENES])
    except (OSError,ValueError,KeyError) as exc:
        return dict(source="rules_fallback",error=type(exc).__name__,metrics=metrics,**propose(rows))


def main():
    torch.set_num_threads(1)
    env=RouteEnv()
    model=SAC.load(OUT/"shared",env=Monitor(env,str(OUT/"llm")),device="cpu")
    model.load_replay_buffer(OUT/"shared_buffer.pkl")
    model.set_random_seed(SEED+1)
    decisions=[]
    for stage in (1,2):
        decision=decide(evaluate(model,700,4))
        decisions.append(dict(stage=stage,**decision))
        (OUT/"llm_decisions.json").write_text(json.dumps(decisions,indent=2))
        env.weights=decision["weights"]
        print(decision,flush=True)
        model.learn(STAGE,reset_num_timesteps=False)
    model.save(OUT/"llm")
    save_csv(OUT/"llm_heldout.csv",evaluate(model,8000,20,score_reference=True))


if __name__=="__main__":
    main()
