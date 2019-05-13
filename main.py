from descriptor.core.eval import evaluate
from descriptor.core.training import train

def main():
    try:
        train()
    except KeyboardInterrupt:
        print('EXIT')

if __name__ == "__main__":
    main()