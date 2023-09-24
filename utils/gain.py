import math

def gain(individual_chaoses, set_sizes, old_total_chaos):
    total_count = 14

    return old_total_chaos - sum([(x/total_count) * y for y, x in zip(individual_chaoses, set_sizes)])

if __name__ == "__main__":

    while True:
        individual_chaoses = [float(x) for x in input("Individual chaos terms: ").strip().split(" ")]
        set_sizes = [int(x) for x in input("Set sizes: ").strip().split(" ")]
        old_total_chaos = float(input("Old total chaos: "))
        print("Gain: ", gain(individual_chaoses, set_sizes, old_total_chaos))
        print()