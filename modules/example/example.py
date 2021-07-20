import random
import time


def main(exp):
    print("RUNNING EXAMPLE MODULE")
    print(f"I will teach you how to multiply numbers!")

    for i in range(5):
        a = random.randint(exp.min_number, exp.max_number)
        b = random.randint(exp.min_number, exp.max_number)
        print(f"{a:<2} * {b:<2} is ", end='')

        for _ in range(3):
            time.sleep(0.5)
            print('.', end='')
        print(f"{a * b}!")


if __name__ == '__main__':
    class exp:
        min_number = 1
        max_number = 10


    main(exp)
