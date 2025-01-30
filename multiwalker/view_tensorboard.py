import os
from tensorboard import program

def launch_tensorboard():
    base_path = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_path, "tensorboard")
    
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir, '--port', '6006'])
    url = tb.launch()
    
    print(f"\n텐서보드가 실행되었습니다.")
    print(f"브라우저에서 다음 주소로 접속하세요: {url}")
    print("종료하려면 Ctrl+C를 누르세요.\n")
    
    try:
        while True:
            input()
    except KeyboardInterrupt:
        print("\n텐서보드를 종료합니다.")

if __name__ == "__main__":
    launch_tensorboard() 