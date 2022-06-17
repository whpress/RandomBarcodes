# single python file for parallel batch processing
# requires: code pickle filename, reads filename, output filename root, segment ID (like "1/2" or "2/2")
from numpy import *
import pickle
import torch
import subprocess
import re
import sys

nargs =  len(sys.argv)
if nargs < 6 :
    print("usage: barcode_batch.py cuda_number segmentID codefilename readsfilename outputrootname")
    raise RuntimeError("bad args in command line")
args = sys.argv

cudano = args[1] # "0"
segmentid = args[2] # "1/25"
codefilename = args[3] # "/home/wpress/randomcode_1e6_34.pkl"
readsfilename = args[4] # "/home/wpress/randomcode_1e6_34_sim_reads"
outputroot = args[5] # "/home/wpress/batch_sim_output"

Ntriage = 10000 # user set to number passed from triage to Levenshtein
Nthresh = 8 # user set to Levenshtein score greater than which is called an erasure

# verify cuda and set device
if torch.cuda.is_available() :
    device = torch.device(f"cuda:{cudano}")
    cudaname = torch.cuda.get_device_name()
else :
    raise RuntimeError("Required GPU not found! Exiting.")
print(f"{segmentid}: Using {device} device {cudaname}.")

# find file size and identify lines in this segment
out = subprocess.Popen(['wc', '-l', readsfilename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
val = out.communicate()[0].split()[0]
if not val.isdigit() : raise RuntimeError(f"failed to count lines in {readsfilename}")
totlines = int(val)

#parse segment ID
num,denom = re.match('([0-9]+)/([0-9]+)',segmentid).groups()
startline = int(totlines*(float(num)-1.)/float(denom))
endline = int(totlines*float(num)/float(denom))
outfilename = outputroot + "_" + num + "of" + denom

# define all functions needed
alphabet = "acgt"
def decode(x): # convert list of ints to string
    s = "".join([alphabet[xx] for xx in x])
    return s
def encode(st): # convert a string into a list of ints
    x = [alphabet.index(ss) for ss in st]
    return x
def seqtomer(seq) : # return list of trimers in a seq
    ans = [int(16*seq[k] + 4*seq[k+1] + seq[k+2]) for k in range(len(seq)-2)]
    return ans
def mertobin(mer) : # trimer list to occupancy uint64 bitmap
    ibin = 0
    for m in mer :
        ibin |= (1 << m)
    return ibin

def makeerrors(seq,srate,irate,drate) : #error mode applied to a sequence seq
    # note: modifies (also returns) seq
    n = len(seq)
    # substitutions
    ns = random.binomial(n,srate*1.3333) # 3/4 of substitutions are "effective"
    ndx = random.randint(low=0,high=n,size=ns)
    vals = random.randint(low=0,high=4,size=ns)
    seq[ndx] = vals
    # deletions
    nd = random.binomial(n,drate)
    ndx = random.randint(low=0,high=n,size=nd)
    seq = delete(seq,ndx)    
    # insertions (at specified rate into smaller seq)
    ni = random.binomial(len(seq),irate)
    ndx = random.randint(low=0,high=len(seq)+1,size=ni)
    vals = random.randint(low=0,high=4,size=ni)
    seq = insert(seq,ndx,vals)
    # pad or truncate to original length
    nn = len(seq)
    if nn > n :
        seq = seq[:n]
    elif nn < n :
        seq = concatenate((seq,random.randint(low=0,high=4,size=n-nn)))
    return seq

m0 = uint64(0x5555555555555555)  # binary: 0101...
m1 = uint64(0x3333333333333333)  # binary: 00110011..
m2 = uint64(0x0f0f0f0f0f0f0f0f)  # binary:  4 zeros,  4 ones ...
m3 = uint64(0x00ff00ff00ff00ff)  # binary:  8 zeros,  8 ones ...
m4 = uint64(0x0000ffff0000ffff)  # binary: 16 zeros, 16 ones ...
m5 = uint64(0x00000000ffffffff)  # binary: 32 zeros, 32 ones
def popcount(x):
# https://github.com/google/jax/blob/6c8fc1b031275c85b02cb819c6caa5afa002fa1d/jax/lax_reference.py#L121-L150
    x = (x & m0) + ((x >>  1) & m0)  # put count of each  2 bits into those  2 bits
    x = (x & m1) + ((x >>  2) & m1)  # put count of each  4 bits into those  4 bits
    x = (x & m2) + ((x >>  4) & m2)  # put count of each  8 bits into those  8 bits
    x = (x & m3) + ((x >>  8) & m3)  # put count of each 16 bits into those 16 bits
    x = (x & m4) + ((x >> 16) & m4)  # put count of each 32 bits into those 32 bits
    x = (x & m5) + ((x >> 32) & m5)  # put count of each 64 bits into those 64 bits
    return x

def allcoses(mer, tcosvecs) : # correlate a mer against all the cosine templates
    ncos = tcosvecs.shape[0]
    cosvec = torch.zeros(ncos, 64, dtype=torch.float, device=device)
    for k in range(ncos) :
        cosvec[k,mer] =  tcoses[k, torch.arange(len(mer), dtype=torch.long, device=device)]
    return torch.sum(torch.unsqueeze(cosvec,dim=1)*tcosvecs,dim=2) # shape [ncos,ngoal_g]

def prank(arr, descending=False) : # returns rank of each element in torch array
    argsrt = torch.argsort(arr, descending=descending)
    rank = torch.zeros(arr.shape, dtype=torch.float, device=device)
    rank[argsrt] = torch.arange(len(argsrt),dtype=torch.float,device=device)
    return rank    

class ApproximateLevenshtein :
    def __init__(s, M, N, Q, zsub, zins, zdel, zskew):
        torch.set_grad_enabled(False) # just in case not done elsewhere!
        s.M = M # length of seq1
        s.N = N # length of each seq2
        s.Q = Q # number of seq2s
        (s.zsub, s.zins, s.zdel, s.zskew) = (zsub, zins, zdel, zskew)
        s.tab = torch.zeros(N+1,Q, device=device)
        
    def __call__(s,seq1,seq2) :
        assert (len(seq1) == s.M) and (seq2.shape[1] == s.N) and (seq2.shape[0] == s.Q)
        s.tab[:,:] = (s.zskew * torch.arange(s.N+1., device=device)).unsqueeze(1) # force broadcast
        for i in range(1,s.M+1) :
            diag = s.tab[:-1,:] + torch.where(seq1[i-1] == seq2.t(), 0., s.zsub) # diagonal move
            s.tab[0,:] += s.zskew
            s.tab[1:,:] += s.zdel # down move
            s.tab[1:,:] = torch.minimum(s.tab[1:,:], diag) # or diag if better
            s.tab[1:,:] = torch.minimum(s.tab[1:,:], s.tab[:-1,:] + s.zins) # right move
            s.tab[1:,:] = torch.minimum(s.tab[1:,:], s.tab[:-1,:] + s.zins) # repeat (>= 0 times) as you can afford
           # N.B.: M >= N gives better approx than N > M, so change arg order accordingly
        return s.tab[s.N,:]

class ParallelLevenshtein :
    def __init__(s, M, N, Q, zsub, zins, zdel, zskew):
        torch.set_grad_enabled(False) # just in case not done elsewhere!
        s.M = M # length of seq1
        s.N = N # length of each seq2
        s.Q = Q # number of seq2s
        (s.zsub, s.zins, s.zdel, s.zskew) = (zsub, zins, zdel, zskew)
        MN1 = M + N + 1
        s.blue = torch.zeros(Q, MN1, MN1, device=device)
        s.bluef = s.blue.view(Q, MN1 * MN1)
        s.ndxr = torch.zeros(M*N, dtype=int, device=device) # index of mer matches array into flat blue
        for m in torch.arange(M,device=device) :
            for n in torch.arange(N,device=device) :
                s.ndxr[n + N*m] = (3*M+2*N+2) + (M+N)*m + (M+N+2)*n
        s.lls = torch.zeros(MN1+1,dtype=torch.int,device=device)       
        s.rrs = torch.zeros(MN1+1,dtype=torch.int,device=device)       
        for i in range(2,MN1+1) :
            s.lls[i] = abs(M - i + 1) + 1
            s.rrs[i] = (M+N-1) - abs(- i + 1 + N )

    def __call__(s, seq1, sseq2): # single seq1, tensor of sseq2s
        assert (len(seq1) == s.M) and (sseq2.shape[1] == s.N) and (sseq2.shape[0] == s.Q)    
        (M1,N1,MN,MN1,MN2) = (s.M + 1, s.N + 1, s.M + s.N, s.M + s.N + 1, s.M + s.N + 2)
        abmatch = (seq1.view(1,s.M,1) != sseq2.view(s.Q,1,s.N)).type(torch.float) * s.zsub
        s.bluef[:,s.ndxr] = abmatch.view(s.Q,s.M*s.N)
        s.bluef[:,torch.arange(s.M,MN2*N1,MN2)] = (s.zskew*torch.arange(N1,device=device)).unsqueeze(0)
        s.bluef[:,torch.arange(s.M,MN*M1,MN)] = (s.zskew*torch.arange(M1,device=device)).unsqueeze(0)
        for k in torch.arange(2,MN1,device=device) :
            ll = s.lls[k]
            rr = s.rrs[k]
            slis = torch.arange(ll,rr+1,2,device=device)
            s.blue[:,k,slis] = torch.minimum(
                s.blue[:,k,slis] + s.blue[:,k-2,slis],
                torch.minimum(
                    s.blue[:,k-1,slis-1] + s.zdel,
                    s.blue[:,k-1,slis+1] + s.zins
                )
            )
        return s.blue[:,-1,s.N]

# load the code from its pickle file
with open(codefilename,'rb') as IN :
    pickledict = pickle.load(IN)
(N, M, allseqs, alltrimers, allbitmaps, coses, cosvecs) = \
    [pickledict[x] for x in ('N', 'M', 'allseqs', 'alltrimers', 'allbitmaps', 'coses', 'cosvecs')]
print(f"{segmentid}: Loaded code with {N} codewords of length {M} from {codefilename}.")

# select the reads for this process
reads = []
with open(readsfilename, 'r') as IN:
    for i, line in enumerate(IN):
        if i < startline :
            pass
        elif i >= endline :
            break
        else :
            reads.append(encode(line[:-1])) # lose the \n
reads = array(reads)

# copy tensors to GPU
torch.set_grad_enabled(False)
tallseqs = torch.tensor(array(allseqs), device=device)
talltrimers = torch.tensor(array(alltrimers), device=device)
tallbitmaps = torch.tensor(allbitmaps.astype(int64), dtype=torch.int64, device=device) # 
tcoses = torch.tensor(coses, dtype=torch.float, device=device)
tcosvecs = torch.tensor(cosvecs, dtype=torch.float, device=device)
print(f"{segmentid}: Found {reads.shape[0]} reads of length {reads.shape[1]}.")

# main decoding steps
Qq = reads.shape[0]
torch.set_grad_enabled(False)

mydist = ApproximateLevenshtein(M,M,Ntriage, 1.,1.,1.,1.)
#mydist = ParallelLevenshtein(M,M,Ntriage, 1.,1.,1.,1.)

Ncos = tcosvecs.shape[0]
dists = torch.zeros(Ncos+1, N, dtype=torch.float, device=device) # will hold distances for each read
cosvec = torch.zeros(Ncos, 64, dtype=torch.float, device=device)
allrank = torch.zeros(Ncos+1 ,N, dtype=torch.float, device=device)
best = torch.zeros(Qq, dtype=torch.long, device=device)

for j,seq in enumerate(reads) :
    # primary and secondary triage
    mer = seqtomer(seq)
    foo = int64(uint64(mertobin(mer))) # need to cast 64 bits to a type known to torch
    seqbin = torch.tensor(foo,dtype=torch.int64,device=device)
    xored = torch.bitwise_xor(seqbin,tallbitmaps)
    dists[0,:] = 64. - popcount(xored) # all Hamming distances
    for k in range(Ncos) :
        cosvec[k,mer] =  tcoses[k, torch.arange(len(mer), dtype=torch.long, device=device)]
    dists[1:,:] = torch.sum(torch.unsqueeze(cosvec,dim=1)*tcosvecs,dim=2) # all cosine distances
    for k in range(Ncos+1) :
        allrank[k,:] = prank(dists[k,:], descending=True) # rank them all
    offset = 1.
    fm = torch.prod(offset+allrank,dim=0)
    fmargsort = torch.argsort(fm)
    # Levenshtein distance
    tseq1 = torch.tensor(seq,device=device)
    tseq2 = tallseqs[fmargsort[:Ntriage],:]
    ans = mydist(tseq1,tseq2)
    ia = torch.argmin(ans) # index into fmargsort of best
    ibest = fmargsort[ia] # index of best codeword in codes
    best[j] = (ibest if ans[ia] <= Nthresh else -1) # erasures returned as -1

with open(outfilename, 'w') as OUT:
    for b in best :        
        OUT.write("%d\n" % b)
print(f"{segmentid}: Decoding results written to {outfilename}. Done.")
