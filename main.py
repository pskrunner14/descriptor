"""Main root module for testing package"""
# from descriptor.core.eval import evaluate
from descriptor.core import train

def main():
    try:
        train()
    except KeyboardInterrupt:
        print('EXIT')

if __name__ == "__main__":
    main()
