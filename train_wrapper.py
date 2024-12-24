import subprocess
import os

def run_training():
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'

    process = subprocess.Popen(
        ['python', 'train.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )

    stdout, stderr = process.communicate()

    return stdout.decode('utf-8'), stderr.decode('utf-8'), process.returncode

def run_evaluation():
    # Assuming eval.sh is in the current directory, adjust the path as necessary
    subprocess.run(['./eval.sh'], check=True)

def main():
    while True:
        stdout, stderr, returncode = run_training()

        if returncode != 0:
            print("Error detected. Retrying...")
        else:
            print("Training completed successfully.")
            break
    
    print("Running evaluation...")
    try:
        run_evaluation()
        print("Evaluation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")

if __name__ == "__main__":
    main()
