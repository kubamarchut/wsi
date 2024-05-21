from train import *
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import multiprocessing


def brain_process(input, layers, input_train, output_train, input_test, output_test, proc, epochs, batch, return_dict, freq,
                  print_plots=True):
    layers_bis = [input.shape[1], layers, 10]  # 10 for num_class
    brain = Brain_network(input_train, output_train, layers_bis)
    model = Brain_network.train_model(brain, input_train, output_train, layers_bis, epochs=epochs, batch_size=batch,
                                      freq=freq)
    eval = Brain_network.evaluate_model(brain, input_test, output_test, model)
    if print_plots:
        plt.plot(brain.probed_epoch, brain.losses)
        plt.title(f"Training Loss, epochs: {epochs}, batch size:{batch}, layers:{layers}")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.grid()
        plt.show()
    key_full = f"E:{epochs}, B:{batch}, L:{layers}"
    return_dict[proc] = [key_full, eval, max(brain.losses), min(brain.losses), epochs, batch, np.std(brain.losses), layers]
    # print(key_full, "-", eval)


def quick_mean(list):
    calc_ls=[]
    for l in list:
        calc_ls.append(l[1])
    sum_ = 0
    for i in calc_ls:
        sum_ += i
    return sum_/len(calc_ls)

def main():

    print_plots = True
    layers = 15

    digits = load_digits()
    input, output = digits.data, digits.target

    output = np.eye(10)[output]

    input_train, input_test, output_train, output_test = train_test_split(
        input, output, test_size=0.3, random_state=10
    )

    scaler = StandardScaler()
    input_train = scaler.fit_transform(input_train)
    input_test = scaler.transform(input_test)

    results = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    proc = 0
    freq = 1000

    for epochs in [100, 200, 500, 1000, 1500, 2000, 2500]:
        for batch in [32, 64, 96, 128]:

            proc += 1
            p = multiprocessing.Process(target=brain_process,
                                            args=(input, layers, input_train, output_train, input_test, output_test, proc, epochs,
                                                  batch, return_dict, freq, print_plots))
            jobs.append(p)
            p.start()
    for p in jobs:
        p.join()

    for e in return_dict.values():
        results.append(e)

    eval_b32 = []
    for t in results:
        if "B:32" in str(t[0]):
            eval_b32.append(t)
    eval_b32_s = sorted(eval_b32, key=lambda x: x[1])
    print("Best batch 32:", eval_b32_s[-1])

    eval_b64 = []
    for t in results:
        if "B:64" in str(t[0]):
            eval_b64.append(t)
    eval_b64_s = sorted(eval_b64, key=lambda x: x[1])
    print("Best batch 64:", eval_b64_s[-1])

    eval_b96 = []
    for t in results:
        if "B:96" in str(t[0]):
            eval_b96.append(t)
    eval_b96_s = sorted(eval_b96, key=lambda x: x[1])
    print("Best batch 96:", eval_b96_s[-1])

    eval_b128 = []
    for t in results:
        if "B:128" in str(t[0]):
            eval_b128.append(t)
    eval_b128_s = sorted(eval_b128, key=lambda x: x[1])
    print("Best batch 128:", eval_b128_s[-1])

    eval_e100 = []
    for t in results:
        if "E:100" in str(t[0]):
            eval_e100.append(t)
    eval_e100_s = sorted(eval_e100, key=lambda x: x[1])
    print("Best epoch 100:", eval_e100_s[-1])

    eval_e200 = []
    for t in results:
        if "E:200" in str(t[0]):
            eval_e200.append(t)
    eval_e200_s = sorted(eval_e200, key=lambda x: x[1])
    print("Best epoch 200:", eval_e200_s[-1])

    eval_e500 = []
    for t in results:
        if "E:500" in str(t[0]):
            eval_e500.append(t)
    eval_e500_s = sorted(eval_e500, key=lambda x: x[1])
    print("Best epoch 500:", eval_e500_s[-1])

    eval_e1000 = []
    for t in results:
        if "E:1000" in str(t[0]):
            eval_e1000.append(t)
    eval_e1000_s = sorted(eval_e1000, key=lambda x: x[1])
    print("Best epoch 1000:", eval_e1000_s[-1])

    eval_e1500 = []
    for t in results:
        if "E:1500" in str(t[0]):
            eval_e1500.append(t)
    eval_e1500_s = sorted(eval_e1500, key=lambda x: x[1])
    print("Best epoch 1500:", eval_e1500_s[-1])

    eval_e2000 = []
    for t in results:
        if "E:2000" in str(t[0]):
            eval_e2000.append(t)
    eval_e2000_s = sorted(eval_e2000, key=lambda x: x[1])
    print("Best epoch 2000:", eval_e2000_s[-1])

    eval_e2500 = []
    for t in results:
        if "E:2500" in str(t[0]):
            eval_e2500.append(t)
    eval_e2500_s = sorted(eval_e2500, key=lambda x: x[1])
    print("Best epoch 2500:", eval_e2500_s[-1])

    print("Mean batch 32:", quick_mean(eval_b32_s))
    print("Mean batch 64:", quick_mean(eval_b64_s))
    print("Mean batch 96:", quick_mean(eval_b96_s))
    print("Mean batch 128:", quick_mean(eval_b128_s))
    print("Mean epoch 100:", quick_mean(eval_e100_s))
    print("Mean epoch 200:", quick_mean(eval_e200_s))
    print("Mean epoch 500:", quick_mean(eval_e500_s))
    print("Mean epoch 1000:", quick_mean(eval_e1000_s))
    print("Mean epoch 1500:", quick_mean(eval_e1500_s))
    print("Mean epoch 2000:", quick_mean(eval_e2000_s))
    print("Mean epoch 2500:", quick_mean(eval_e2500_s))



    results_sorted = sorted(results, key=lambda x: (x[4], x[5], x[7]))
    plt.figure(figsize=(16, 9))
    for r in results_sorted:
        plt.errorbar(r[0], r[1], fmt='o', capthick=2, capsize=5)
    plt.title(f"Training Evaluations with F1 scores")
    plt.xlabel("Epochs,Batch size")
    plt.ylabel("Final evaluation")
    plt.xticks(rotation=-60)
    plt.grid()
    plt.show()

    plt.figure(figsize=(16, 9))
    for r in results_sorted:
        plt.errorbar(r[0], r[2], r[6], fmt='o', capthick=2, capsize=5)
    plt.title(f"Training Max Loss")
    plt.xlabel("Epochs,Batch size")
    plt.ylabel("Loss")
    plt.xticks(rotation=-60)
    plt.grid()
    plt.show()

    plt.figure(figsize=(16, 9))
    for r in results_sorted:
        plt.errorbar(r[0], r[3], r[6], fmt='o', capthick=2, capsize=5)
    plt.title(f"Training Min Loss")
    plt.xlabel("Epochs,Batch size")
    plt.ylabel("Loss")
    plt.xticks(rotation=-60)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
