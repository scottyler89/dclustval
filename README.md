# README #

### What is this repository for? ###

dclustval: Dense Cluster Validation
When you have two distance matrices (independent measures of the same observations) and tentative cluster labels, you can use dclustval (standing for dense cluster validation) to determine whether the distances observed in the training dataset were recapitulated in the validation dataset.

This repository is in particular designed to work with count splitting on matrices of observations made from a Poisson sampling process (although the distributions can be count data, not necessarily Poisson distributed only). We have implemented a separate repository for bootstrapped count splitting (rather than distributional thinning):
`python3 -m pip install count_split` # automatically installed in the requirements here

Note that if you use our count splitting repository, you should also site the original authors who came up with the count-splitting idea:
https://arxiv.org/abs/2207.00554

There's a tutorial below, but if you want function level documentation, feel free to head over to [Read The Docs!](https://dclustval.readthedocs.io/en/latest/index.html)


### How do I get set up? ###

`python3 -m pip install dclustval`

You can also install using the setup.py script in the distribution like so:
`python3 setup.py install`
or cd into the repository directory and:
`python3 -m pip install .`

## How do I use this package? ###

### Trivial example ###
Really all we need is two distance matrices that are matched in their samples, so 2 independent measures of the same samples/observations.

In reality, this is intended for count-splitting, but just so you get the jist, here's a trivial example that doesn't use count-splitting:
```python
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as euc
from dclustval.cluster import do_cluster_validation
np.random.seed(123456)
n_obs = 400
n_features = 2
dist1 = euc(np.random.random(size=(n_obs,n_features)))
dist2 = euc(np.random.random(size=(n_obs,n_features)))
# The cluster labels should be zero indexed numpy arrays (the sklearn standard)
bad_labels = np.array([0 for _ in range(int(n_obs)/2)]+[1 for _ in range(int(n_obs)/2)])
stat_mat, p_mat_adj, final_labels = do_cluster_validation(dist1, dist2, bad_labels)
print(set(final_labels))
print(stat_mat)
print(p_mat_adj)
```
You should see something that looks like this:
```python
>>> stat_mat, p_mat_adj, final_labels = do_cluster_validation(dist1, dist2, np.array(bad_labels))
initial comps_list: [[0, 1]]
>>> print(set(final_labels))
{0} # Now we have only 1 cluster!
>>> print(stat_mat)
# this is the matrix of t-statistics for judging if the clusters are significantly different from each other
[[0.         0.81536625]
 [0.81536625 0.        ]]
>>> print(p_mat_adj)
# this is their adjusted p-values. Locations match the input cluster label indices (row/col 0 means cluster 0)
[[0.         0.41486594]
 [0.41486594 0.        ]]

```

## A not so trivial example ##
For this demo, we're also going include differing depth across cells as a factor, so
we'll also install the downsampling package and some plotting with seaborn:
`python3 -m pip install bio-pyminer-norm seaborn`
The other dependencies come within this package!

Here we'll set up some (perhaps elaborate) dummy data
```python
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from copy import deepcopy
from count_split.count_split import multi_split
from pyminer_norm.downsample import new_rewrite_get_transcript_vect as ds
from pyminer_norm.downsample import downsample_mat
from sklearn.metrics.pairwise import euclidean_distances as euc
from anticor_features.anticor_stats import no_p_spear
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from dclustval.cluster import do_cluster_validation
from anticor_features.anticor_features import get_anti_cor_genes
## first make some dummy single cell data
# I'm making absolutely no claim that this is a "good" simulation of 
# single cell data, just a quick whip-up of something that looks somewhat similar
np.random.seed(123456)
n_genes = 5000
n_group_specific_genes = 500
n_groups = 2
# make sure we have enough genes for this sim
assert n_group_specific_genes*n_groups < n_genes
cells_per_group = 500
n_cells = int(n_groups*cells_per_group)
# We'll make two clusters, but first we'll make a "base" transcriptome
base_transcriptome_lambdas = np.random.negative_binomial(.1,.01,size=n_genes)
group_transcriptome_lambdas = []
min_lambda = 0.1
group_vect = []
## make some cell-type specific gene choices
cell_type_genes = []
for g in range(1,n_groups+1):
    cell_type_genes.append(np.arange((g-1)*n_group_specific_genes,(g)*n_group_specific_genes))

## make the main transcript lambdas for each group
for g in range(n_groups):
    temp_lamb=deepcopy(base_transcriptome_lambdas)
    temp_lamb[cell_type_genes[g]]=np.random.negative_binomial(.1,.01,size=len(cell_type_genes[g]))
    ## make sure we're at noise level for other group's genes
    for g2 in range(n_groups):
        if g!=g2:
            temp_lamb[cell_type_genes[g2]]=min_lambda
    # add low level non-specific noise for zeros
    temp_lamb[temp_lamb<min_lambda]=min_lambda
    group_transcriptome_lambdas.append(temp_lamb)
    # also add the group membership vector
    group_vect += [g for i in range(cells_per_group)]


# here cells are in rows, genes are in columns
X = np.zeros((n_cells,n_genes))
for cell_idx in range(n_cells):
    temp_group = group_vect[cell_idx]
    temp_transcript_vect = np.zeros((n_genes))
    for t in range(n_genes):
        # Add a poisson sample for each gene's lambda +noise for their given group
        temp_transcript_vect[t]=np.random.poisson(max(min_lambda,group_transcriptome_lambdas[temp_group][t]))
    X[cell_idx,:]=deepcopy(temp_transcript_vect)


# If you want to check out how insane the depth effect is, even with nice distributions
# Set this to True and see how badly over-clustered things can appear without depth
# normalization!
def depth_norm(in_mat,target=None):
    # Normalizes the row
    loading = np.sum(in_mat,axis=1)
    if target is None:
        # Default to counts/avg total counts
        target = np.mean(loading)
    loading_factors = loading/target
    return(in_mat / loading_factors[:,None])
    

def tenk_norm(in_mat):
    return(depth_norm(in_mat, target=10000))


checkout_depth = False
if checkout_depth:
    depth_vect = np.random.lognormal(np.log(2500),1, size=n_cells)
    Xdepth=deepcopy(X)
    for cell_idx in range(n_cells):
        Xdepth[cell_idx,:]=ds(X[cell_idx,:],int(depth_vect[cell_idx]))#, 4000)#
    Xdepth = Xdepth.T
else:
    # For the "main event" we'll account for depth through downsampling though
    # We'll downsample to 2500 counts per cell
    Xdepth = downsample_mat(X.T,2500).todense()


```

Okay - so that was all just setting up the data for analysis...

Now we'll get to the part that's pacticularly relevant to this repository!


```python
## Note that if you're using real data with anndata
# (this is what underlies scanpy), this will be under the "X" key
# X = adata.X.T
# Also note that count_split assumes that genes are in rows and cells are in
# the columns, so we have to give it the transpose


## Now we'll split X into 2 matrices by count_splitting
X1, X2 = multi_split(Xdepth,percent_vect=[.5,.5])
# The returned matrices are int64, so we'll convert them back
# and also re-transpose to make sure that the cells are in the rows
X1 = X1.T.astype(float)
X2 = X2.T.astype(float)


# Now we'll normalize to counts per avg total counts
# How to do this is still being debated...
def pc50_euc(temp_mat, npcs=50, count_norm = 10000, **kwargs):
    ## do the counts per 10k & log
    n_cells = temp_mat.shape[1]
    cglsums = np.squeeze(np.array(np.sum(temp_mat,axis=0)))
    norm_factor = cglsums/count_norm
    for i in range(n_cells):
        temp_mat[:,i]/=norm_factor[i]
    num_comps = min(npcs,int(n_cells/2),int(temp_mat.shape[0]/2)) 
    pca = PCA(n_components=num_comps)
    if "todense" in dir(temp_mat):
        temp_mat.data = np.log2(1+temp_mat.data)
    else:
        temp_mat = np.log2(1+temp_mat)
        temp_mat[np.isnan(temp_mat)]=0
    if "toarray" in dir(temp_mat):
        if type(temp_mat)==csc_matrix:
            pass
        else:
            temp_mat=csc_matrix(temp_mat)
        ## to save on memory, remove the non-expressed genes
        keep_idxs = sorted(list(set(temp_mat.indices)))
        out_pcs = pca.fit_transform(temp_mat[keep_idxs,:].toarray().T)
    else:
        out_pcs = pca.fit_transform(temp_mat.T)
    euc_mat = euc(out_pcs)
    return(euc_mat)


# This also expects cells in columns, so feed it the transpose
X1_feats = get_anti_cor_genes(np.log2(1+X1.T), ["gene_"+str(i) for i in range(n_genes)])
X1_select_feat_idxs=np.where(X1_feats['selected']==True)[0]
X2_feats = get_anti_cor_genes(np.log2(1+X2.T), ["gene_"+str(i) for i in range(n_genes)])
X2_select_feat_idxs=np.where(X2_feats['selected']==True)[0]

X1dist = pc50_euc(X1[:,X1_select_feat_idxs].T.astype(float))
X2dist = pc50_euc(X2[:,X2_select_feat_idxs].T.astype(float))

# just to see what it looks like
sns.clustermap(np.log2(1+X1[:,X1_select_feat_idxs]))
plt.show()


# Now we'll waaay over-cluster it
kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(deepcopy(X1dist))
kmeans.labels_

stat_mat, p_mat_adj, final_labels = do_cluster_validation(X1dist, X2dist, kmeans.labels_)
# initial comps_list: [[0, 2, 3, 5, 6, 7], [8, 1, 4, 9]]

print(set(final_labels))
# {0, 1}
```
For some reason one of the steps isn't using the set seeds, so the results are coming out slightly variable. That being said, above we can see that the original 10 clusters are merged down to 2 (or sometimes 3 b/c of the seed issue).

Now we'll check out what the training distances look like, with the original over-clustered results and the 
```python
# Now if we look at the training distances, using the final labels as 
unique_labels = np.unique(final_labels+kmeans.labels_.tolist())
color_palette = sns.color_palette("Set1", len(unique_labels))

sns.clustermap(X1dist, col_colors=np.array(color_palette)[final_labels], row_colors=np.array(color_palette)[kmeans.labels_])
plt.show()

```
Below is the image showing over clustering, that get's fixed by `do_cluster_validation` from `dclustval.cluster`
![Distance Matrix and Annotations](https://github.com/scottyler89/dclustval/raw/main/assets/original_vs_dclustval_merged.png)

## What this repository doesn't do! ##

This approach _only_ tells you whether the distances are recapitulated between two data splits. What it *does not* tell you is that your final clusters are biologically meaningful. They could be reproducibly different because of technical effects too. So still bring a health dose of caution to the interpretation of the clusters on the other side of using this repository!

As a testimate to that, let's look at the results if we *did* use the "check out depth" option, giving cells a fairly tight log-normal distribution of total counts:
```python
>>> stat_mat, p_mat_adj, final_labels = do_cluster_validation(X1dist, X2dist, kmeans.labels_)
initial comps_list: [[0], [1, 9], [2], [3, 7], [4], [5], [6], [8]]
>>> 
>>> print(set(final_labels))
{0, 1, 2, 3, 4, 5, 6, 7}
```
While depth is still a factor, we can end up being under the incorrect impression that these clusters are different from each other & meaningfully so, despite the fact that they are indeed reproducibly different, but for technical reasons! This needs to be fixed in the upstream processing 

And here's the heatmap showing the distances and final cluster results:

![No Norm with Depth Distance Matrix and Annotations](https://github.com/scottyler89/dclustval/raw/main/assets/depth_original_vs_dclustval_merged.png)

See how insane these detph effects are?? They'll still also look reproducibly different between your splits if you don't account for this depth effect. Note however that this normalization process is still highly debated.

And no, unfortunately, counts per 10k doesn't fix the issue =/
```python
>>> stat_mat, p_mat_adj, final_labels = do_cluster_validation(X1dist, X2dist, kmeans.labels_)
initial comps_list: [[0], [1], [2], [3, 7], [4], [5], [6], [8], [9]]
>>> 
>>> print(set(final_labels))
{0, 1, 2, 3, 4, 5, 6, 7, 8}
>>> 
```
![10k Distance Matrix and Annotations](https://github.com/scottyler89/dclustval/raw/main/assets/depth_10k_original_vs_dclustval_merged.png)


Counts per mean total counts doesn't do the trick either:
```python
>>> stat_mat, p_mat_adj, final_labels = do_cluster_validation(X1dist, X2dist, kmeans.labels_)
initial comps_list: [[0], [1], [2, 7], [3], [9, 4], [5], [6], [8]]
>>> 
>>> print(set(final_labels))
{0, 1, 2, 3, 4, 5, 6, 7}
```
![Mean Depth Distance Matrix and Annotations](https://github.com/scottyler89/dclustval/raw/main/assets/depth_mean_count_original_vs_dclustval_merged.png)


The issue is that the depth effect creates a non-Euclidean covariate... Difficult... Downsampling is safe for this, but is often seen as 'agressive.'

