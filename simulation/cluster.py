import numpy as np


class Cluster:
    def __init__(self, vc_dict, num_gpus_per_node, num_cpus_per_node):
        self._vc_dict = vc_dict
        self._num_gpus_per_node = num_gpus_per_node
        self._num_cpus_per_node = num_cpus_per_node
        self.vc_num = len(vc_dict)
        self.node_num = sum(vc_dict.values())
        self.vc_list = []
        self.init_cluster_vc()
        self.total_gpus = sum(vc.total_gpus for vc in self.vc_list)
        self.total_cpus = sum(vc.total_cpus for vc in self.vc_list)

    def init_cluster_vc(self):
        for k, v in self._vc_dict.items():
            vc = VC(k, v, self._num_gpus_per_node, self._num_cpus_per_node)
            self.vc_list.append(vc)

    def cluster_free_gpus(self):
        return sum(vc.vc_free_gpus() for vc in self.vc_list)

    def cluster_free_cpus(self):
        return sum(vc.vc_free_cpus() for vc in self.vc_list)


# Virtual Cluster
class VC:
    def __init__(self, vc_name, node_num, num_gpus_per_node, num_cpus_per_node):
        self.vc_name = vc_name
        self.node_num = node_num
        self.base_node_num = node_num
        self._num_gpus_per_node = num_gpus_per_node
        self._num_cpus_per_node = num_cpus_per_node
        self.node_list = []
        self.init_vc_node()
        self.total_gpus = num_gpus_per_node * node_num
        self.total_cpus = num_cpus_per_node * node_num

        self.colocate_enable = 0
        # Temp Node num with additional temp_node_num_base
        self.temp_node_num_base = 9999
        self.has_temp_node = False  # To avoid first-time scaling error

    def init_vc_node(self):
        for i in range(self.node_num):
            node = Node(i, self._num_gpus_per_node, self._num_gpus_per_node)
            self.node_list.append(node)

    def check_node_inside_vc(self, node_id):
        for i in self.node_list:
            if i.node_name == node_id:
                return True
        return False

    def check_node_inside_idle_vc(self, node_id):
        idle_list = self.idle_node_list()
        for i in idle_list:
            if i.node_name == node_id:
                return True
        return False

    def add_new_node(self, change_node_num, force_same_node):
        for i in range(change_node_num):
            temp_node_num = i + self.temp_node_num_base
            if self.check_node_inside_vc(temp_node_num) and force_same_node:
                # temp_node_num = temp_node_num + 1000
                # raise ValueError("Temp node num already exists")
                return False
            node = Node(temp_node_num, self._num_gpus_per_node, self._num_gpus_per_node)
            self.node_list.append(node)
        self.node_num = self.node_num + change_node_num
        self.total_gpus = self._num_gpus_per_node * self.node_num
        self.total_cpus = self._num_cpus_per_node * self.node_num

    def exchange_node_status(self, idle_node, i):
        # Just for simple simulation implementation in some rare cases.
        # In reality, we can directly remove different nodes.
        assert idle_node.check_free_gpus() == self._num_gpus_per_node
        temp_node = self.get_node(self.temp_node_num_base + i)
        temp_node.update_node_name(idle_node.node_name)
        idle_node.update_node_name(self.temp_node_num_base + i)
        temp_node.exchange_job_status()

    def remove_idle_node(self, change_node_num, force_same_node):
        idle_node_list = self.idle_node_list()
        if len(idle_node_list) < abs(change_node_num):
            return False  # Not enough idle nodes
        idle_node_list.sort(key=lambda x: x.node_name, reverse=True)
        idle_node_list = idle_node_list[: abs(change_node_num)]
        for i in range(abs(change_node_num)):
            if idle_node_list[i].node_name < self.temp_node_num_base and force_same_node and self.has_temp_node:
                self.exchange_node_status(idle_node_list[i], i)
                idle_node_list = self.idle_node_list()
                idle_node_list.sort(key=lambda x: x.node_name, reverse=True)
                assert idle_node_list[0].node_name >= self.temp_node_num_base
            to_remove_node = idle_node_list[i]
            self.node_list.remove(to_remove_node)
        self.has_temp_node = True
        assert len(self.node_list) == self.node_num + change_node_num
        self.node_num = self.node_num + change_node_num
        self.total_gpus = self._num_gpus_per_node * self.node_num
        self.total_cpus = self._num_cpus_per_node * self.node_num
        return True

    def update_vc_node(self, change_node_num, force_same_node=True):
        if change_node_num > 0:
            self.add_new_node(change_node_num, force_same_node)
        elif change_node_num < 0:
            self.remove_idle_node(change_node_num, force_same_node)
        else:
            raise ValueError("`change_node_num` should not be 0")

    def get_node(self, node_id):
        # node = self.node_list[node_id]
        # assert node.node_name == node_id
        # return node

        for i in self.node_list:
            if i.node_name == node_id:
                return i

    def vc_free_gpus(self):
        return sum(node.free_gpus for node in self.node_list)

    def vc_free_cpus(self):
        return sum(node.free_cpus for node in self.node_list)

    def idle_node_list(self):
        idle_node_list = []
        for node in self.node_list:
            if node.free_gpus == self._num_gpus_per_node:
                idle_node_list.append(node)
        return idle_node_list

    def avail_node_list(self):
        avail_node_list = []
        for node in self.node_list:
            if node.free_gpus > 0:
                avail_node_list.append(node)
        return avail_node_list

    def release_resource(self, job):
        nodes_list = job["nodes"]
        for dict in nodes_list:
            for i, gpu_list in dict.items():
                node = self.get_node(i)
                assert node.node_name == i
                assert node.release_gpu(gpu_list, job)
        return True

    def check_vc_colocate_jobs(self, job):
        # nodes_list = job["nodes"]
        # recover_jobs = set()
        # for dict in nodes_list:
        #     for i, gpu_list in dict.items():
        #         node = self.node_list[i]
        #         jobs = node.check_colocate_jobs(gpu_list, job)
        #         recover_jobs |= set(jobs)
        # return list(recover_jobs)
        nodes_list = job["nodes"]
        dict = nodes_list[0]
        for i, gpu_list in dict.items():
            node = self.get_node(i)
            colo_job_id = node.check_colocate_jobs(gpu_list, job)
            if colo_job_id:
                return colo_job_id
            else:
                raise NotImplementedError

    # Only one job running in a node
    def consolidate_node_num(self):
        list = []
        for node in self.node_list:
            if node.job_num == 1:
                list.append(node)
        return len(list)

    def shared_node_num(self):
        list = []
        for node in self.node_list:
            if node.job_num > 1:
                list.append(node)
        return len(list)

    def check_vc_sm_util(self):
        list = []
        for node in self.node_list:
            list.append(node.check_avg_gpu_util())
        return np.mean(list)

    def check_vc_gmem_util(self):
        list = []
        for node in self.node_list:
            list.append(node.check_avg_mem_util())
        return np.mean(list)

    def check_vc_active_sm_util(self):
        list = []
        for node in self.node_list:
            util = node.check_active_avg_gpu_util()
            if util:
                list.append(util)
        if list:
            return np.mean(list)
        else:
            return 0

    def check_vc_active_gmem_util(self):
        list = []
        for node in self.node_list:
            util = node.check_active_avg_mem_util()
            if util:
                list.append(util)
        if list:
            return np.mean(list)
        else:
            return 0


class Node:
    def __init__(self, node_name, num_gpus, num_cpus):
        self.node_name = node_name
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        self.previous_node_name = node_name

        self.job_num = 0
        self.free_cpus = num_cpus
        # self.free_gpus = num_gpus
        # self.colocate_gpu_num = 0

        self.node_job_dict = {}
        self.node_gpu_dict = self.init_gpu_dict()
        self.node_gutil = self.init_gpu_state()  # GPU Utilization
        self.node_gmem = self.init_gpu_state()  # GPU Memory Usage

    @property
    def free_gpus(self):
        return self.check_free_gpus()

    def init_gpu_dict(self):
        gdict = {}
        for i in range(self.num_gpus):
            gdict.update({i: []})
        return gdict

    def init_gpu_state(self):
        gdict = {}
        for i in range(self.num_gpus):
            gdict.update({i: 0})
        return gdict

    def check_avg_gpu_util(self):
        return np.mean(list(self.node_gutil.values()))

    def check_avg_mem_util(self):
        return np.mean(list(self.node_gmem.values()))

    def check_active_avg_gpu_util(self):
        gutils = list(self.node_gutil.values())
        active_ls = list(filter(lambda x: x > 0, gutils))
        if active_ls:
            return np.mean(active_ls)
        else:
            return False

    def check_active_avg_mem_util(self):
        gmems = list(self.node_gmem.values())
        active_ls = list(filter(lambda x: x > 0, gmems))
        if active_ls:
            return np.mean(active_ls)
        else:
            return False

    def check_free_gpus(self):
        count = 0
        for k, v in self.node_gpu_dict.items():
            if v == []:
                count += 1
        return count

    def check_free_gpu_list(self):
        free_list = []
        for k, v in self.node_gpu_dict.items():
            if v == []:
                free_list.append(k)
        return free_list

    def check_colocate_gpu_list(self):
        co_gpus = []
        for k, v in self.node_gpu_dict.items():
            if len(v) == 2:
                co_gpus.append(k)
        return co_gpus

    def check_colocate_jobs(self, gpu_list, job):
        # colocate_jobs = set()
        # for i in gpu_list:
        #     for j in self.node_gpu_dict[i]:
        #         if j is not job:
        #             colocate_jobs.add(j)
        # return list(colocate_jobs)
        # colocate_jobs = set()
        for k, v in self.node_job_dict.items():
            if v == gpu_list and k != job:
                return k
        return None

    """colocate usage"""

    def allocate_colocate_gpu(self, gpu_list, job, gutil, gmem):
        # num_gpu = len(gpu_list)
        # self.colocate_gpu_num += num_gpu
        self.job_num += 1

        for i in gpu_list:
            assert len(self.node_gpu_dict[i]) == 1
            self.node_gpu_dict[i].append(job)
            self.node_gutil[i], self.node_gmem[i] = gutil, gmem
        self.node_job_dict.update({job["job_id"]: gpu_list})
        return True

    """allocate"""

    def allocate_gpu(self, num_gpu, job):
        assert num_gpu <= self.free_gpus
        # self.free_gpus -= num_gpu
        self.job_num += 1
        allocate_gpus = []
        toallocate = num_gpu
        for k, v in self.node_gpu_dict.items():
            if toallocate == 0:
                break
            if v == []:
                allocate_gpus.append(k)
                self.node_gpu_dict[k].append(job)
                self.node_gutil[k] = job["gpu_util"]
                self.node_gmem[k] = job["gmem"]
                toallocate -= 1
        assert num_gpu == len(allocate_gpus)
        self.node_job_dict.update({job["job_id"]: allocate_gpus})
        return allocate_gpus

    """release"""

    def release_gpu(self, gpu_list, job):
        if job["exclusive"] > 0:
            assert self.free_gpus + len(gpu_list) <= self.num_gpus
            # self.free_gpus += len(gpu_list)
            self.job_num -= 1
        else:
            # assert self.colocate_gpu_num >= num_gpu
            # self.colocate_gpu_num -= len(gpu_list)
            self.job_num -= 1

        for i in gpu_list:
            assert isinstance(i, int)
            self.node_gpu_dict[i].remove(job)
            if self.node_gpu_dict[i] == []:
                self.node_gutil[i] = 0
                self.node_gmem[i] = 0
            else:
                assert len(self.node_gpu_dict[i]) == 1
                exist_job = self.node_gpu_dict[i][0]
                self.node_gutil[i] = exist_job["gpu_util"]
                self.node_gmem[i] = exist_job["gmem"]

        self.node_job_dict.pop(job["job_id"])

        return True

    def update_node_name(self, new_name):
        # Echo `exchange_node_status`
        self.previous_node_name = self.node_name
        self.node_name = new_name

    def exchange_job_status(self,):
        # Echo `exchange_node_status`
        jobs = []
        for k, v in self.node_gpu_dict.items():
            if v != []:
                for job in v:
                    if job not in jobs:
                        jobs.append(job)
        for job in jobs:
            for allocate_dict in job["nodes"]:
                k, v = list(allocate_dict.items())[0]
                if k == self.previous_node_name:
                    new_dict = {self.node_name: v}
                    job["nodes"].remove(allocate_dict)
                    job["nodes"].append(new_dict)

    # Future Extension
    def allocate_cpu(self, num_cpu):
        if num_cpu > self.free_cpus:
            return False
        else:
            self.free_cpus -= num_cpu
            return True

    def release_cpu(self, num_cpu):
        assert self.free_cpus + num_cpu <= self.num_cpus
        self.free_cpus += num_cpu
        return True


if __name__ == "__main__":
    # test = Node(1, 8, 96)
    # print(test.node_gpu_dict)
    # job = {"n": 1, "m": 2}
    # ls = []
    # ls.append(job)
    # for j in ls:
    #     if j is job:
    #         print(1)
    # colocate_jobs = set()
    # print(list(colocate_jobs))
    # target_nodes = [
    #     {1: [0, 1, 2, 3]},
    #     {2: [0, 1, 2, 3, 4, 5, 6, 7]},
    #     {4: [0, 1, 2, 3, 4,]},
    #     {5: [0, 1, 2, 3, 4, 5, 6, 7]},
    # ]
    # # nodes = sorted(target_nodes, key=lambda x: len(list(x.values())[0]), reverse=True)
    # # print(nodes)
    # alloc_nodes = []
    # for node_dict in target_nodes:
    #     alloc_nodes.append((list(node_dict.keys())[0], list(node_dict.values())[0]))
    # print(alloc_nodes)
    # # print(len(list(d.values())[0]))
    # # print(len(d.values()[0]))

    # recover_jobs = set()
    # jobs1 = ["aaa", "ddd"]
    # jobs2 = ["bbb", "ccc", "aaa"]
    # recover_jobs |= set(jobs1)
    # recover_jobs |= set(jobs2)
    # print(list(recover_jobs))
    # import random

    # job1 = {"n": 1, "m": 2}
    # job2 = {"n": 2, "m": 2}
    # jobs = [job1, job2]
    # print(np.mean(list(job1.values())))
    gutil = [0]
    y = list(filter(lambda x: x > 0, gutil))

