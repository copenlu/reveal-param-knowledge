import json
import argparse, pickle
import torch
import numpy as np
import glob, os
import pdb

from collections import Counter
from scipy.stats import zscore


class NeuronAnalysis():
    def __init__(self, args):

        self.train_data = self._get_data(args.neuron_folder, mode="train")
        print("# of training data:", len(self.train_data))

        if args.dataset == "csqa":
            self.dev_data = self._get_data(args.neuron_folder, mode="validation")
        else:
            self.dev_data = self._get_data(args.neuron_folder, mode="test")
        print("# of dev data:", len(self.dev_data))

        self.threshold = args.threshold
        self.map_mode = args.map_mode
        self.args = args

        self.cutoff = 1


    def convert_inst(self, inf_result):

        new_inf_result = {}
        for train_idx, dev_scores in inf_result.items():
            for dev_idx, score in enumerate(dev_scores):
                new_inf_result.setdefault(dev_idx, [])
                new_inf_result[dev_idx].append(score)

        print("Converted ...")

        return new_inf_result
    # end of convert_inst

    # April 3rd, 2024 - after ARR Feb review
    def ia_neurons_compare(self):

        important_nr_from_na = self.summary_important_neuron()

        inf_results = pickle.load(open(self.args.inf_result, "rb"))

        if len(inf_results) != len(self.dev_data):
            inf_results = self.convert_inst(inf_results)

        important_nr_from_ia = []
        for dev_idx, train_ranks in inf_results.items():

            train_top_1 = np.argsort(train_ranks)[::-1][0]
            nr_info = self.train_data[train_top_1]

            # top-3
            imp_nr_indices = torch.argsort(torch.tensor(nr_info["attributions"]), descending=True)[:100]
            # top-1
            imp_nr = ["%d-%d" % (nr_info["neurons"][i][0], nr_info["neurons"][i][1]) for i in imp_nr_indices][:self.cutoff]

            if isinstance(imp_nr, str):
                important_nr_from_ia.append(imp_nr)
            else:
                important_nr_from_ia += imp_nr

        # compare important_nr_from_na <> important_nr_from_ia
        from_na = Counter(important_nr_from_na)
        from_ia = Counter(important_nr_from_ia)
        print("NA:", len(from_na))
        print("IA:", len(from_ia))

        print("Overlap:", len(set(important_nr_from_na)&set(important_nr_from_ia)))
    # end of ia_neurons_compare

    # April 3rd, 2024 - after ARR Feb review
    def summary_important_neuron(self):

        top_1_neurons_list = []

        for d in self.dev_data:
            # top-3
            dev_nr_indices = torch.argsort(torch.tensor(d["attributions"]), descending=True)[:100]
            # top-1
            top_nr = ["%d-%d" % (d["neurons"][i][0], d["neurons"][i][1]) for i in dev_nr_indices][:self.cutoff]

            if isinstance(top_nr, str):
                top_1_neurons_list.append(top_nr)
            else:
                top_1_neurons_list += top_nr

        return top_1_neurons_list

    def sort_neuron(self, nr_info):

        imp_nr_indices = torch.argsort(torch.tensor(nr_info["attributions"]), descending=True)
        imp_nr = ["%d-%d" % (nr_info["neurons"][i][0], nr_info["neurons"][i][1]) for i in imp_nr_indices]
        return imp_nr


    # April 4th, after review
    def neurons_from_na_instances(self):

        na_instances_result = json.load(open("./result_map_ver2/neuron_mapping_mnli_opt125m.json"))
        inf_results = pickle.load(open(self.args.inf_result, "rb"))

        if len(inf_results) != len(self.dev_data):
            inf_results = self.convert_inst(inf_results)

        important_nr_na = []
        important_nr_ia = []
        top_instances_ia = []
        top_instances_na = []
        for i in range(len(inf_results)):
            ia_ranks = inf_results[i]
            na_ranks = na_instances_result[str(i)]

            sorted_ia_instances = np.argsort(ia_ranks)[::-1].tolist()
            sorted_na_instances = np.argsort(na_ranks)[::-1].tolist()


            ia_top_1 = sorted_ia_instances[0] # instance
            na_top_1 = sorted_na_instances[0] # instance

            ia_nr_info = self.train_data[ia_top_1]
            na_nr_info = self.train_data[na_top_1]

            top_instances_ia += sorted_ia_instances[:10]
            top_instances_na += sorted_na_instances[:10]

            # neurons
            ia_sorted_nr = self.sort_neuron(ia_nr_info)
            na_sorted_nr = self.sort_neuron(na_nr_info)

            important_nr_ia.append(ia_sorted_nr[0])
            important_nr_na.append(na_sorted_nr[0])

        # compare important_nr_from_na <> important_nr_from_ia
        from_ia = Counter(important_nr_ia)
        from_na = Counter(important_nr_na)
        print(len(from_na))
        print(len(from_ia))
        pdb.set_trace()



    def mapping_instances(self):

        retrieved_trains = {}
        total_retrieved = []

        for dev_idx, d in enumerate(self.dev_data):

            # dev offset = 5
            dev_nr_indices = torch.argsort(torch.tensor(d["attributions"]), descending=True)[:5]
            dev_nr_list = ["%d-%d" %(d["neurons"][i][0], d["neurons"][i][1]) for i in dev_nr_indices]

            train_scores = []
            for train_idx, train_d in enumerate(self.train_data):
                # train offset = 10
                train_nr_indices = np.argsort(train_d["attributions"])[::-1]
                train_nr_list = ["%d-%d" %(train_d["neurons"][i][0], train_d["neurons"][i][1]) for i in train_nr_indices]
                train_nr_attr = zscore([train_d["attributions"][i] for i in train_nr_indices])

                gain_list = []
                discount_list = []
                for rank, nr in enumerate(dev_nr_list):
                    if nr in train_nr_list:
                        train_nr_rank = train_nr_list.index(nr)
                        gain_list.append(train_nr_attr[train_nr_rank])
                        discount_list.append(train_nr_rank+2.0)

                    else:
                        gain_list.append(0.0)
                        discount_list.append(2.0)


                per_dcg_score = (np.exp2(gain_list) -1) / (np.log2(discount_list))
                dcg_score = np.sum(per_dcg_score)

                if dcg_score == 0.0:
                    train_scores.append(-np.inf)
                else:
                    train_scores.append(dcg_score)

            retrieved_trains[str(dev_idx)] = train_scores
            sorted = np.argsort(train_scores)[::-1].tolist()
            total_retrieved += sorted[:10]

        print("# of unique influential instances:", len(set(total_retrieved)))

        print("Check, %d in dict and %d of testdata" %(len(retrieved_trains), len(self.dev_data)))


        return retrieved_trains


    def _get_data(self, neuron_folder, mode="train"):
        # train

        # for mode in mode_list:
        each_list = []
        files = glob.glob("%s/neuron_%s_*.json" % (neuron_folder, mode))
        # print(files)

        # sorting
        key = "neuron_%s_" %mode
        before_sort = {int(f.split(key)[-1].replace(".json", "")) : f for f in files}
        sorted_key = sorted(before_sort)

        for tf in sorted_key:
            print(before_sort[tf])
            tmp_t = json.load(open(before_sort[tf]))
            each_list += tmp_t

        return each_list


def ranking_instances(filename):

    mapping_result = json.load(open(filename))

    maximum_rank = len(mapping_result["0"])

    new_inf_dict = {}

    for test_idx, na_scores in mapping_result.items():

        na_rank = np.argsort(na_scores)[::-1]

        for rank_id, na_train_idx in enumerate(na_rank):
            new_inf_dict.setdefault(na_train_idx, [])

            if na_scores[na_train_idx] == -np.inf:
                new_inf_dict[na_train_idx].append(maximum_rank)
            else:
                new_inf_dict[na_train_idx].append(rank_id)

    inf_rank_sum = {}
    for train_idx, rank_list in new_inf_dict.items():
        inf_rank_sum[int(train_idx)] = float(np.sum(rank_list))

    return inf_rank_sum


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--neuron_folder", type=str, default="./results_neuron/neurons_th_0.1_1020")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--map_mode", type=str, default="top-1")
    parser.add_argument("--dataset", type=str, default="averitec")
    parser.add_argument("--na_inf_outfile", type=str)
    parser.add_argument("--mapping_outfile", type=str)
    parser.add_argument("--inf_result", type=str, default="../influence-func/fc-influ-func/mnli_opt125m/inf_func.pickle")

    args = parser.parse_args()

    print(args)

    filename = args.neuron_folder.split("/")[-1]

    analysis_module = NeuronAnalysis(args)

    analysis_module.ia_neurons_compare()

    # analysis_module.neurons_from_na_instances()

    # if os.path.exists("./result_map_ver3/%s.json" %filename):
    #     print("Exists, create instance full ranking")
    #     inf_rank_sum = ranking_instances("./result_map_ver3/%s.json" %filename)
    #     json.dump(inf_rank_sum, open("../data/%s/na_%s_ver3.json" %(args.dataset, filename), "w"), indent=4)

    # else:
    # mapping = analysis_module.mapping_instances()
    # json.dump(mapping, open("./result_map_ver3/%s.json" %filename, "w"), indent=4)
