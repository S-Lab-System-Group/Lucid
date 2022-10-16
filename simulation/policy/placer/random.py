class RandomPlacement:
    def __init__(self, vc):
        self.vc = vc
        self.name = "random"
        self.avail_nodes = self.vc.avail_node_list()

    """Random placement"""

    def update_avail_nodes(self):
        self.avail_nodes = self.vc.avail_node_list()

    def randomSelect(self, job_gpu_num):
        self.update_avail_nodes()
        alloc_nodes = []

        for node in self.avail_nodes:
            if node.free_gpus < job_gpu_num:
                alloc_nodes.append((node, node.free_gpus))
                job_gpu_num -= node.free_gpus
                continue
            else:
                alloc_nodes.append((node, job_gpu_num))
                return True, alloc_nodes
        return False, alloc_nodes

    def place(self, job):
        vc_free_gpu_num = self.vc.vc_free_gpus()
        job_gpu_num = job["gpu_num"]

        # Total Free GPU Check
        if vc_free_gpu_num < job_gpu_num:
            return False

        select_flag, alloc_nodes = self.randomSelect(job_gpu_num)

        """ Placement """
        if select_flag:
            for (node, req_gpu) in alloc_nodes:
                allocate_gpus = node.allocate_gpu(req_gpu, job)
                job["nodes"].append({node.node_name: allocate_gpus})
            return True
        else:
            return False
