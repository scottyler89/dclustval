import os
import numpy as np
import seaborn as sns
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances as euc
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind, rankdata
#from count_split.count_split import split_mat_counts, split_mat_counts_h5

## Hack to work-around dependency issues
import  scipy.signal.signaltools
def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

scipy.signal.signaltools._centered = _centered
##
from statsmodels.stats.multitest import fdrcorrection

##################################################################################
def dense_rank(in_vect):
    return(rankdata(in_vect,method="dense"))

# This function takes two input vectors, concatenates them, applies the dense rank 
# function to the concatenated vector, and then splits the output back into two vectors.
# Dense rank is used for handling ties in the data, assigning equal rank to equal values.
def dense_rank_both(in_vect1, in_vect2):
    """
    Performs a dense rank operation on two input vectors.

    Parameters
    ----------
    in_vect1 : ndarray
        The first input vector.
    in_vect2 : ndarray
        The second input vector.

    Returns
    -------
    out_vect1 : ndarray
        The dense-ranked version of the first input vector.
    out_vect2 : ndarray
        The dense-ranked version of the second input vector.
    """
    # Concatenate the input vectors and rank them.
    out_vect = dense_rank(np.concatenate((in_vect1, in_vect2)))
    # Split the ranked vector into two based on the original input vector lengths.
    out_vect1 = out_vect[:in_vect1.shape[0]]
    out_vect2 = out_vect[in_vect1.shape[0]:]
    return(out_vect1, out_vect2)


# This function takes a temporary list of cell labels, finds unique labels and 
# then returns a dictionary with the labels as keys and the indices where the 
# label occurs in the list as values.
def catelogue_labs(temp_cell_labels):
    """
    Catalogs the indices of unique labels in a list.

    Parameters
    ----------
    temp_cell_labels : list
        The list of cell labels.

    Returns
    -------
    out_dict : dict
        A dictionary mapping each unique label to a list of indices at which the label occurs.
    """
    # Get all unique labels.
    all_labs = list(set(temp_cell_labels))
    cell_labs = np.array(temp_cell_labels)
    out_dict = {}
    # For each label, find indices where it occurs.
    for lab in all_labs:
        out_dict[lab] = np.where(cell_labs == lab)[0].tolist()
    return(out_dict)


# This function finds cliques in the given graph and removes the nodes of the 
# clique with the highest average edge weight.
def get_weighted_cliques(G):
    """
    Finds and removes the highest-weight clique in a network graph.

    Parameters
    ----------
    G : NetworkX graph
        The input network graph.

    Returns
    -------
    winner_clique : list
        The highest-weight clique in the input graph.
    G : NetworkX graph
        The input graph with the highest-weight clique removed.
    """
    # Find all cliques in the graph.
    all_cliques = nx.find_cliques(G)
    clique_list = []
    all_avg_weights = []
    # For each clique, calculate the average edge weight and remove the nodes of the highest-weight clique.
    # This corresponds to the highest (least significant) p-value clique
    for clique in all_cliques:
        clique_list.append(clique)
        sub_g = G.subgraph(clique).copy()
        temp_weights = []
        for edge in sub_g.edges():
            print("\t\t",edge)
            temp_weights.append(sub_g[edge[0]][edge[1]]["weight"])
        all_avg_weights.append(np.mean(np.array(temp_weights)))
        print("\t",all_avg_weights[-1])
    highest_avg_weight_idx = np.argmax(all_avg_weights)
    winner_clique = clique_list[highest_avg_weight_idx]
    for win in winner_clique:
        G.remove_node(win)
    return(winner_clique, G)


# This function recursively finds and removes the highest-weight cliques until the 
# graph has no nodes left. The removed cliques are returned as a list.
def get_recursive_cliques(G):
    """
    Finds and removes highest-weight cliques recursively from a network graph.

    Parameters
    ----------
    G : NetworkX graph
        The input network graph.

    Returns
    -------
    final_out_mergers : list
        A list of all highest-weight cliques removed from the graph.
    """

    final_out_mergers = []
    while len(G.nodes())>0:
        temp_comp, G = get_weighted_cliques(G)
        final_out_mergers.append(temp_comp)
    return(final_out_mergers)


# This function creates a network graph based on the two input vectors and a vector of 
# p-values. Each edge in the graph connects two corresponding elements from the two input 
# vectors and has a weight equal to the corresponding p-value. The function then finds 
# and removes highest-weight cliques recursively and returns the removed cliques as a list.
def get_merged_clusters(first, second, p):
    """
    Creates a network graph and recursively finds and removes highest-weight cliques.

    Parameters
    ----------
    first : ndarray
        The first input vector.
    second : ndarray
        The second input vector.
    p : ndarray
        A vector of p-values, each corresponding to a pair of elements in the input vectors.

    Returns
    -------
    final_merged_clusters : list
        A list of all highest-weight cliques removed from the graph.
    """
    output_comps = []
    # Create a graph with edges defined by the input vectors and weights defined by the p-values.
    G=nx.Graph(directed=False)
    for i in range(len(first)):
        G.add_edge(first[i],second[i],weight=p[i])
    # Recursively find and remove highest-weight cliques.
    final_merged_clusters = get_recursive_cliques(G)
    return(final_merged_clusters)


# This function takes a list of components and an adjacency matrix of p-values.
# It pairs each component with every other component and stores their p-value.
# It then orders the component pairs by their p-value, creates a cluster merging 
# order, and returns a list of merged clusters, with each cluster represented as a 
# list of its component labels.
def get_ordered_list_by_p(comp, p_mat_adj):
    """
    Generates a list of merged clusters by ordering component pairs by their p-values.

    Parameters
    ----------
    comp : list
        A list of components.
    p_mat_adj : ndarray
        An adjacency matrix of p-values.

    Returns
    -------
    merged_comps_list : list
        A list of merged clusters, with each cluster represented as a list of its component labels.
    """
    # sort the components
    comp = sorted(comp)
    first = []
    second = []
    p = []
    # Pair each component with every other component and store their p-value.
    for i in range(len(comp)):
        temp_clust_1 = comp[i]
        for j in range(len(comp)):
            if i != j:
                temp_clust_2 = comp[j]
                first.append(temp_clust_1)
                second.append(temp_clust_2)
                p.append(p_mat_adj[temp_clust_1, temp_clust_2])
    # Order the component pairs by their p-value.
    clust_order = np.argsort(p)[::-1]
    first = np.array(first)[clust_order]
    second = np.array(second)[clust_order]
    p = np.array(p)[clust_order]
    # Merge clusters based on the created order.
    merged_comps_list = get_merged_clusters(first, second, p)
    all_clusts = list(set(first.tolist()+second.tolist()))
    included_dict = {clust:False for clust in all_clusts}
    # Mark included clusters as True
    for comp in merged_comps_list:
        for single_clust in comp:
            included_dict[single_clust]=True
    # For clusters not included in the merge, append them separately
    for key, value in included_dict.items():
        if value is False:
           merged_comps_list.append([key]) 
    return(merged_comps_list)


# This function takes a list of components and an adjacency matrix of p-values.
# For each component in the list, if it contains more than two elements, it orders 
# the component pairs by their p-value and merges them into clusters.
def finalize_comp_list(comps_list, p_mat_adj):
    """
    Finalizes a list of components by merging component pairs into clusters based on their p-values.

    Parameters
    ----------
    comps_list : list
        A list of components.
    p_mat_adj : ndarray
        An adjacency matrix of p-values.

    Returns
    -------
    final_comp_list : list
        A list of finalized components.
    """
    final_comp_list = []
    for comp in comps_list:
        if len(comp)>2:
            final_comp_list += get_ordered_list_by_p(comp, p_mat_adj)
        else:
            final_comp_list.append(comp)
    return(final_comp_list)


# This function takes a list of components and returns a matrix that represents 
# the adjacency of the components. The size of the matrix is determined by the 
# maximum component label plus one. Each row represents a component and each 
# column represents a potential adjacent component.
def comp_to_mat(comp_list):
    """
    Converts a list of components into an adjacency matrix.

    Parameters
    ----------
    comp_list : list
        A list of components.

    Returns
    -------
    out_mat : ndarray
        A matrix that represents the adjacency of the components.
    """
    temp_clusts = -1
    for comp in comp_list: temp_clusts = max(temp_clusts, max(comp))
    out_mat = np.zeros((temp_clusts+1,temp_clusts+1))
    for comp in comp_list:
        for element in comp:
            out_mat[element,np.array(comp)]=1
    return(out_mat)


# This function takes temporary cell labels, a significance matrix, and an adjacency 
# matrix of p-values. It creates a network graph from the significance matrix, 
# finds its connected components, catalogues the original labels, orders and merges 
# the component pairs based on their p-value, and returns the final labels for each 
# cell in the original label list.
def get_final_labels(temp_cell_labels, sig_mat, p_mat_adj):
    """
    Generates the final labels for each cell.

    Parameters
    ----------
    temp_cell_labels : list
        A list of temporary cell labels.
    sig_mat : ndarray
        A significance matrix.
    p_mat_adj : ndarray
        An adjacency matrix of p-values.

    Returns
    -------
    final_labels : list
        The final labels for each cell.
    """
    # Create a network graph from the significance matrix
    try:
        G = nx.from_numpy_matrix(sig_mat)
    except:
        G = nx.from_numpy_array(sig_mat)
    # Find the connected components
    comps = nx.connected_components(G)
    comps_list = []
    for comp in comps: comps_list.append(list(comp))
    print("initial comps_list:",comps_list)
    # Catalogue the original labels
    idx_lookups = catelogue_labs(temp_cell_labels)
    comp_idxs = []
    num_comps = len(comps_list)
    for comp_idx in range(len(comps_list)):
        comp_idxs.append([])
        for orig_lab in comps_list[comp_idx]:
            comp_idxs[comp_idx]+=idx_lookups[orig_lab]
        comp_idxs[comp_idx]=sorted(comp_idxs[comp_idx])
    # Determine the final labels of the components
    final_labels = np.zeros(len(temp_cell_labels))
    for final_clust in range(num_comps):
        temp_idxs = comp_idxs[final_clust]
        final_labels[temp_idxs]=final_clust
    return(final_labels)


# This function performs cluster validation for the input distance matrices
# and temporary cell labels. It tests if the average distance within each cluster 
# is less than the average distance between clusters and returns a statistic and 
# p-value matrix that represents the significance of the difference for each 
# cluster pair, as well as the final cell labels after potential cluster merging. 
# If plot_dir is provided, it also saves the distribution plots for each cluster pair 
# and the resulting matrices to the specified directory.
def do_cluster_validation(mat_1_dist, 
                        mat_2_dist, 
                        temp_cell_labels, 
                        alpha=0.01, 
                        plot_dir = "",
                        validation_merge = True):
    """
    Performs cluster validation and potentially merges clusters.

    Parameters
    ----------
    mat_1_dist : ndarray
        The first distance matrix.
    mat_2_dist : ndarray
        The second distance matrix.
    temp_cell_labels : list
        A list of temporary cell labels.
    alpha : float, optional
        The significance level, default is 0.01.
    plot_dir : str, optional
        The directory to save the plots, default is an empty string.
    validation_merge : bool, optional
        Whether to perform cluster merging, default is True.

    Returns
    -------
    stat_mat : ndarray
        The statistic matrix for each cluster pair.
    p_mat_adj : ndarray
        The adjusted p-value matrix for each cluster pair.
    final_labels : list
        The final labels for each cell.
            
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics.pairwise import euclidean_distances as euc
    >>> from dclustval.cluster import do_cluster_validation
    >>> np.random.seed(123456)
    >>> n_obs = 400
    >>> n_features = 2
    >>> dist1 = euc(np.random.random(size=(n_obs,n_features)))
    >>> dist2 = euc(np.random.random(size=(n_obs,n_features)))
    >>> bad_labels = np.array([0 for _ in range(int(n_obs)/2)]+[1 for _ in range(int(n_obs)/2)])
    >>> stat_mat, p_mat_adj, final_labels = do_cluster_validation(dist1, dist2, bad_labels)
    """
    # Get a list of all unique clusters
    all_clusts = sorted(list(set(temp_cell_labels.tolist())))
    # Initialize matrices for storing the statistics and p-values
    stat_mat = np.zeros((len(all_clusts),len(all_clusts)))
    p_mat = np.zeros((len(all_clusts),len(all_clusts)))
    # If validation_merge is False, return empty matrices and the original labels
    if not validation_merge:
        return(stat_mat, p_mat+1, temp_cell_labels)
    # For each cluster, calculate the within and between cluster distances
    for i in range(len(all_clusts)):
        temp_clust_1_idxs = np.sort(np.where(temp_cell_labels==all_clusts[i])[0])
        for j in range(len(all_clusts)):
            # Get the indices of the cells in each cluster
            if i==j:
                # If i equals j, only calculate the within cluster distances
                temp_sub_mat_1 = mat_1_dist[temp_clust_1_idxs,:]
                temp_sub_mat_1 = temp_sub_mat_1[:,temp_clust_1_idxs]
                mat_1_dist_flat =temp_sub_mat_1[np.triu_indices(temp_sub_mat_1.shape[0], k=1)]
                temp_sub_mat_2 = mat_2_dist[temp_clust_1_idxs,:]
                temp_sub_mat_2 = temp_sub_mat_2[:,temp_clust_1_idxs]
                mat_2_dist_flat = temp_sub_mat_2[np.triu_indices(temp_sub_mat_2.shape[0], k=1)]
            else:
                # If i doesn't equal j, calculate both the within and between cluster distances
                temp_clust_2_idxs = np.sort(np.where(temp_cell_labels==all_clusts[j])[0])
                mat_1_dist_flat = mat_1_dist[temp_clust_1_idxs,:]
                mat_1_dist_flat = mat_1_dist_flat[:,temp_clust_2_idxs].flatten()
                mat_2_dist_flat = mat_2_dist[temp_clust_1_idxs,:]
                mat_2_dist_flat =mat_2_dist_flat[:,temp_clust_2_idxs].flatten()
                ## now get within group1
                temp_sub_mat_1 = mat_1_dist[temp_clust_1_idxs,:]
                temp_sub_mat_1 = temp_sub_mat_1[:,temp_clust_1_idxs]
                within1_mat_1_dist_flat =temp_sub_mat_1[np.triu_indices(temp_sub_mat_1.shape[0], k=1)]
                temp_sub_mat_2 = mat_2_dist[temp_clust_1_idxs,:]
                temp_sub_mat_2 = temp_sub_mat_2[:,temp_clust_1_idxs]
                within1_mat_2_dist_flat = temp_sub_mat_2[np.triu_indices(temp_sub_mat_2.shape[0], k=1)]
                ## now get within group2
                temp_sub_mat_1 = mat_1_dist[temp_clust_2_idxs,:]
                temp_sub_mat_1 = temp_sub_mat_1[:,temp_clust_2_idxs]
                within2_mat_1_dist_flat =temp_sub_mat_1[np.triu_indices(temp_sub_mat_1.shape[0], k=1)]
                temp_sub_mat_2 = mat_2_dist[temp_clust_2_idxs,:]
                temp_sub_mat_2 = temp_sub_mat_2[:,temp_clust_2_idxs]
                within2_mat_2_dist_flat = temp_sub_mat_2[np.triu_indices(temp_sub_mat_2.shape[0], k=1)]
                ####################
                if plot_dir != "":
                    plt.clf()
                    comparison_name = str(i)+"_vs_"+str(j)
                    fig, axes = plt.subplots(1, 2)
                    sns.distplot(within1_mat_1_dist_flat, ax=axes[0],color="grey")
                    sns.distplot(within2_mat_1_dist_flat, ax=axes[0],color="grey")
                    sns.distplot(mat_1_dist_flat, ax=axes[0],color="red")
                    sns.distplot(within1_mat_2_dist_flat, ax=axes[1],color="grey")
                    sns.distplot(within2_mat_2_dist_flat, ax=axes[1],color="grey")
                    sns.distplot(mat_2_dist_flat, ax=axes[1],color="red")
                    axes[1].set_ylabel('')
                    out_plot = os.path.join(plot_dir,comparison_name+".png")
                    fig.suptitle(comparison_name)
                    if not os.path.isdir(plot_dir):
                        os.mkdir(plot_dir)
                    plt.savefig(out_plot, dpi=600, bbox_inches='tight')
                    plt.clf()
                within_pop1_rank, across_pops_rank = dense_rank_both(within1_mat_2_dist_flat, mat_2_dist_flat)
                stat_1, p_1 = ttest_ind(within_pop1_rank, across_pops_rank)
                within_pop2_rank, across_pops_rank = dense_rank_both(within2_mat_2_dist_flat, mat_2_dist_flat)
                stat_2, p_2 = ttest_ind(within_pop1_rank, across_pops_rank)
                statistic = min(stat_1, stat_2)
                p_val = max(p_1, p_2)
                stat_mat[i,j]=statistic
                stat_mat[j,i]=statistic
                p_mat[i,j]=p_val
                p_mat[j,i]=p_val
    # get the FDR corrected significance matrices
    sig_mat, p_mat_adj = fdrcorrection(p_mat.flatten(), alpha=alpha)
    p_mat_adj = p_mat_adj.reshape(p_mat.shape)
    sig_mat = p_mat_adj>alpha
    # Use these p-values to get the final labels
    final_labels = get_final_labels(temp_cell_labels, sig_mat, p_mat_adj)
    final_labels = [int(f) for f in final_labels]
    if plot_dir != "":
        ##
        plt.clf()
        out_plot = os.path.join(plot_dir,"rank_t_statistic.png")
        sns.clustermap(stat_mat)
        plt.savefig(out_plot, dpi=600, bbox_inches='tight')
        ##
        plt.clf()
        out_plot = os.path.join(plot_dir,"p_adj.png")
        sns.clustermap(p_mat_adj)
        plt.savefig(out_plot, dpi=600, bbox_inches='tight')
        ##
        plt.clf()
        out_plot = os.path.join(plot_dir,"significance_bool.png")
        sns.clustermap(p_mat_adj>alpha)
        plt.savefig(out_plot, dpi=600, bbox_inches='tight')
    return(stat_mat, p_mat_adj, final_labels)

