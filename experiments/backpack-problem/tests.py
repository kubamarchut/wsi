def get_n_in_bit_range(bits):
    return [bin(i)[2:].zfill(bits) for i in range(2**bits)]


def cost(vector, data):
    sum = 0

    for index, item in enumerate(data):
        sum += int(int(vector[index]) * int(item[0]))

    return -sum


def object_function(vector, data):
    sum = 0

    for index, item in enumerate(data):
        sum += int(int(vector[index]) * int(item[1]))

    return sum


def get_all_possabilities(data):
    bits = len(data)
    vectors = get_n_in_bit_range(bits)

    for vector in vectors:
        print(f"{cost(vector, data):>4}", end=" | ")

    print()

    for vector in vectors:
        print(f"{object_function(vector, data):>4}", end=" | ")

    # print(vectors)
