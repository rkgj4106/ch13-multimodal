import argparse
from pathlib import Path
from engine import plan,make_video,Storyboard

if __name__=="__main__":
    p=argparse.ArgumentParser(description="Generate moving footage using Gemma and Wan")
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--script",type=Path)
    group.add_argument("--storyboard",type=Path)
    p.add_argument("--scenes",type=int,choices=range(1,7),default=1)
    p.add_argument("--frames",type=int,choices=[49,81],default=49)
    p.add_argument("--steps",type=int,choices=range(20,51),default=30)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--voice",choices=["none","gtts"],default="none")
    p.add_argument("--output",type=Path,default=Path("output"))
    args=p.parse_args()
    board=Storyboard.model_validate_json(args.storyboard.read_text(encoding="utf-8")) if args.storyboard else plan(args.script.read_text(encoding="utf-8"),args.scenes)
    print(make_video(board,args.output.resolve(),args.voice,args.frames,args.steps,args.seed,
                     lambda status,msg:print(f"[{status}] {msg}",flush=True)))
