import numpy as np
from sigpyproc.Readers import FilReader
from sigpyproc.Header import Header

def coadd(fdirs, outname, weights=None, bulk=50000):
    """
    incoherently add sigproc filterbank files

    Parameters
    ----------
    fdirs: array_like
        filenames of filterbanks to read
    outname : str
        filename to which data is saved
    weights : array_like
        weighing factors for each filterbank file.
        defaults to equal weights
    bulk : int, optional
        number of samples to read at a time
        default: 50000
    """
    # Instantiate filterbanks
    fils = []
    for fname in fdirs:
        tmp = FilReader(fname)
        assert tmp.header.nbits = [8,32], "Can only take 8 or 32-bit format filterbanks"
        fils.append(tmp)

    nbits = tmp.header.nbits

    if not weights:
        weights = np.ones((len(fdirs)))
        weights /= np.sum(weights)

    assert len(fdirs) == len(weights), "fdirs and weights should have the same length"
    assert sum(weights) < 1.0001, "Weights should add up to 1"
    for i in range(1,len(fdirs)):
        assert fils[i-1].header.nchans == fils[i].header.nchans, "Input FBanks should have the same nchans"


    # Determine start and end of output
    tstart = max([fil.header.tstart for fil in fils])
    tstop = [fil.header.tstart + fil.header.nsamples*fil.header.tsamp/(60.*60*24) for fil in fils]

    # Determine start sample for every input filterbank
    start_sample = []
    for i,fil in enumerate(fils):
        start_sample.append(round((tstart - fil.header.tstart)*24*60*60/fil.header.tsamp))
    total_samples = int(np.floor(min([(tstop[i] - tstart)*24*60.*60/fils[i].header.tsamp for i in range(len(fils))])))
    nchans = fils[0].header.nchans

    # Define the output header
    outheader = Header(fils[0].header)
    outheader.tstart = tstart
    outheader.nsamples = total_samples

    # Write header to file in binary format
    outfile = open(outname,"wb")
    outfile.write(outheader.SPPHeader())

    # Read data from filterbanks, weigh it, and write to output in binary format
    samples_left = total_samples
    output = np.zeros((nchans,bulk))
    while samples_left > 0:
        samples_to_read = min(samples_left,bulk)
        if output.shape[1] != samples_to_read:
            output = np.zeros((nchans,samples_to_read))
        else:
            output.fill(0)

        for i,fil in enumerate(fils):
            block = fil.readBlock(start_sample[i]+total_samples-samples_left,samples_to_read)
            output += weights[i]*block

        samples_left -= samples_to_read

        if nbits == 8:
            output = output.astype('uint8').transpose()
        elif nbits == 32:
            output = output.astype('float32').transpose()
        output.tofile(outfile)

def test():
    import glob
    UTC = "20170803075011"
    BEAM = 1
    BASE_DIR = "/lustre/projects/p002_swin/askap/craft_tmp/SB00154/"+UTC

    utcs = glob.glob(str(BASE_DIR+"/*/*/*.%02d.fil") %BEAM)

    weights = np.ones((len(utcs)))
    weights /= np.sum(weights)

    coadd(utcs,"./test_out.fil",weights)


def main(args):
    nfiles = len(args.infiles)
    if nfiles < 2:
        print "Specify at least 2 files, exiting..."
        sys.exit(0)
    if args.weights:
        assert len(args.weights) == nfiles, "Input the same number of weights as input files"
        weights = [float(i) for i in args.weights]
    else:
        weights = np.ones(nfiles)
        weights /= np.sum(weights)

    coadd(args.infiles,args.outfile,weights,args.ngulp)

if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(description = "Adds incoherently filterbanks")
    
    parser.add_argument('-f',type=str,nargs='+',required=True,
        help='Input sigproc files',dest='infiles')
    parser.add_argument('-w',type=list,nargs='+',required=False,
            help='weights',dest='weights')
    parser.add_argument('-o',type=str,required=True,
            help='output name',dest='outfile')
    parser.add_argument('-n',type=int,required=False,
            help='Number of samples to read at a time (default: 50000)',
            dest='ngulp',default=50000)

    args = parser.parse_args()
    main(args)
