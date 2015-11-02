import matplotlib
import numpy as np;
matplotlib.use('Agg')
import matplotlib.pyplot as plt;

def writeHTML(file_name,im_paths,captions,height=200,width=200):
    f=open(file_name,'w');
    html=[];
    f.write('<!DOCTYPE html>\n');
    f.write('<html><body>\n');
    f.write('<table>\n');
    for row in range(len(im_paths)):
        f.write('<tr>\n');
        for col in range(len(im_paths[row])):
            f.write('<td>');
            f.write(captions[row][col]);
            f.write('</td>');
            f.write('    ');
        f.write('\n</tr>\n');

        f.write('<tr>\n');
        for col in range(len(im_paths[row])):
            f.write('<td><img src="');
            f.write(im_paths[row][col]);
            f.write('" height='+str(height)+' width='+str(width)+'"/></td>');
            f.write('    ');
        f.write('\n</tr>\n');
        f.write('<p></p>');
    f.write('</table>\n');
    f.close();

def createScatterOfDiffsAndDistances(diffs,title,xlabel,ylabel,out_file,dists=None):
    plt.figure();
    plt.title(title);
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);

    print out_file

    diffs_all=diffs.ravel();
    dists_all=[];
    if dists is None:
        dists_all=np.arange(diffs.shape[1]);
        dists_all=np.repeat(dists_all,diffs.shape[0]);
        # for idx in range(len(diffs)):
        #     dists_all.extend(range(1,len(diffs[idx])));
    else:
        dists_all=np.ravel(dists)
        # for dist in dists:
        #     dists_all.extend(dist);
    # bins=(max(diffs_all)-min(diffs_all),max(diffs_all)-min(diffs_all));
    heatmap, xedges, yedges = np.histogram2d(dists_all,diffs_all,bins=(100,45))
    heatmap=heatmap.T
    print heatmap.shape
    # print bins

    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    plt.imshow(heatmap)
    plt.savefig(out_file);
    plt.close();


def createImageAndCaptionGrid(img_paths,gt_labels,indices,text_labels):
    im_paths=[[[] for i in range(indices.shape[1]+1)] for j in range(indices.shape[0])]
    captions=[[[] for i in range(indices.shape[1]+1)] for j in range(indices.shape[0])]
    for r in range(indices.shape[0]):
        im_paths[r][0]=img_paths[r];
        captions[r][0]='GT class \n'+text_labels[gt_labels[r]]+' '+str(gt_labels[r]);
        for c in range(indices.shape[1]):
            pred_idx=indices[r][c]
            im_paths[r][c+1]=img_paths[pred_idx];
            if gt_labels[pred_idx] !=gt_labels[r]:
                captions[r][c+1]='wrong \n'+text_labels[gt_labels[pred_idx]]+' '+str(gt_labels[pred_idx]);
            else:
                captions[r][c+1]='';
    return im_paths,captions


def plotDistanceHistograms(diffs_curr,degree,out_file,title='',xlabel='Distance Rank',ylabel='Frequency',delta=0,dists_curr=None,bins=10,normed=False):
    
    if dists_curr is None:
        dists_curr=np.array(range(1,diffs_curr.shape[1]+1));
        dists_curr=np.expand_dims(dists_curr,0);
        dists_curr=np.repeat(dists_curr,diffs_curr.shape[0],0);

    # diffs_to_analyze=[0,45,90,135,180];
    # plt.ion();
    # for diff_to_analyze in diffs_to_analyze:
    diffs=diffs_curr-degree;
    diffs=abs(diffs);
    idx=np.where(diffs<=delta)
    dists=dists_curr[idx[0],idx[1]];

    plt.figure();
    print  'len(dists)',len(dists);
    plt.hist(dists,bins,normed=normed);
    plt.title(title);
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    plt.savefig(out_file);
    plt.close();

def plotErrorBars(dict_to_plot,x_lim,y_lim,xlabel,y_label,title,out_file,margin=[0.05,0.05],loc=2):
    
    plt.title(title);
    plt.xlabel(xlabel);
    plt.ylabel(y_label);
    
    if y_lim is None:
        y_lim=[1*float('Inf'),-1*float('Inf')];
    
    max_val_seen_y=y_lim[1]-margin[1];
    min_val_seen_y=y_lim[0]+margin[1];
    print min_val_seen_y,max_val_seen_y
    max_val_seen_x=x_lim[1]-margin[0];
    min_val_seen_x=x_lim[0]+margin[0];
    handles=[];
    for k in dict_to_plot:
        means,stds,x_vals=dict_to_plot[k];
        
        min_val_seen_y=min(min(np.array(means)-np.array(stds)),min_val_seen_y);
        max_val_seen_y=max(max(np.array(means)+np.array(stds)),max_val_seen_y);
        
        min_val_seen_x=min(min(x_vals),min_val_seen_x);
        max_val_seen_x=max(max(x_vals),max_val_seen_x);
        
        handle=plt.errorbar(x_vals,means,yerr=stds);
        handles.append(handle);
        print max_val_seen_y
    plt.xlim([min_val_seen_x-margin[0],max_val_seen_x+margin[0]]);
    plt.ylim([min_val_seen_y-margin[1],max_val_seen_y+margin[1]]);
    plt.legend(handles, dict_to_plot.keys(),loc=loc)
    plt.savefig(out_file);
    plt.close();

def plotSimple(xAndYs,out_file,title='',xlabel='',ylabel='',legend_entries=None,loc=0):
    plt.title(title);
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    # assert len(xs)==len(ys)
    handles=[];
    for x,y in xAndYs:
        handle,=plt.plot(x,y);
        handles.append(handle);
    if plt.legend is not None:
        plt.legend(handles,legend_entries,loc=loc)
    plt.savefig(out_file);
    plt.close();    


def main():
    print 'hello';

if __name__=='__main__':
    main();