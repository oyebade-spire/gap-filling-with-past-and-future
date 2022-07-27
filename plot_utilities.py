from cProfile import label
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd

# #############increase plot resolution
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['savefig.dpi'] = 300

# def plot_data(data, plot_title=None):
#     """ function: plots the given data
#     data: data to be plotted """
#     if plot_title !=None:
#         plt.title(plot_title)
#     plt.imshow(data, cmap='RdBu')
#     plt.colorbar()
#     plt.show()


def create_colormap(file_in):
    
    def rgb_to_hex(r,g,b):
        c = '#%02x%02x%02x' % (r, g, b)   
        return c

    # Read file
    colnames=['value', 'R', 'G', 'B', 'A', 'label'] 
    data = pd.read_csv(file_in, names=colnames, sep=",", skiprows=[0,1])
    colors = []
    # Create hex colors based on RGB colors in text file 
    for index, row in data.iterrows():
        colors.append(rgb_to_hex(int(row['R']),int(row['G']),int(row['B'])))
    # Create map    
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    return cmap


def plot_data(data, plot_title=None, sv_plt=False, sv_path=None, sv_idx=None, custom_cmap=None):
    """ function: plots the given data
    data: data to be plotted
    plot_title: optional overall caption for the two plots
    sv_plt: whether to save plotted figure 
    sv_path: path for saving plots
    sv_idx: index used for saving plotted figure
    custom_cmap: customized cmap for plotting """

    plt.figure(dpi=300)
    if plot_title !=None:
        plt.title(plot_title)
    if custom_cmap ==None:
        custom_cmap='RdBu'

    plt.imshow(data, vmin=0, vmax=0.5, cmap=custom_cmap)
    plt.colorbar()

    if sv_plt== True:
        # plt.imsave(sv_path + 'gap-filling result for day' + ' ' + str(sv_idx)  + '.png', fig)
        plt.savefig(sv_path + 'gap-filling result for day' + ' ' + 
        str(sv_idx)  + '.png', bbox_inches='tight')
    plt.figure(dpi=300)
    plt.show()


def plot_abs_error(data, plot_title=None, sv_plt=False, sv_path=None, 
    sv_idx=None, custom_cmap =None):
    """ function: plots the given data
    data: data to be plotted
    plot_title: optional overall caption for the two plots
    sv_plt: whether to save plotted figure 
    sv_path: path for saving plots
    sv_idx: index used for saving plotted plt.figure """

    plt.figure(dpi=300)
    if plot_title !=None:
        plt.title(plot_title)
    
    cmap_nw = plt.get_cmap('YlGn').copy()
    cmap_nw.set_bad(color='orange')

    if custom_cmap ==None:
        cmap_nw='RdBu'

    plt.imshow(data, cmap=cmap_nw)
    plt.colorbar()

    if sv_plt== True:
        # plt.imsave(sv_path + 'gap-filling result for day' + ' ' + str(sv_idx)  + '.png', fig)
        plt.savefig(sv_path + 'Error image for' + ' ' + 
        str(sv_idx)  + '.png', bbox_inches='tight')
    plt.show()
    # if show_plot== True or show_plot== None:
    #     plt.show()
    # else:
    #     plt.close(fig)




def plot_scatter(x1, x2, lb_1=None, lb_2=None, y_lbl=None, x_lbl=None, plt_title=None):
    """ function: makes a scatter plot of the ground-truth data and gap-filled data
    x1: ground-truth data
    x2: gap-filled data 
    lb_1: plot label for x1 
    lb_2: plot label for x2 
    y_lbl: y-axis label
    x_lbl: x-axis label 
    plot_title: overall title for plot  """
    
    plt.figure(dpi=300)
    plt.scatter(np.arange(x1.shape[0]), x1, label = lb_1, linewidth=0.05, marker= ".")   #plot series as scatter
    plt.scatter(np.arange(x2.shape[0]), x2, label = lb_2, linewidth=0.05, marker= ".")   #plot series as scatter
    if y_lbl !=None:
        plt.ylabel(y_lbl)
    if x_lbl !=None:
        plt.xlabel(x_lbl)
    if plt_title !=None:
        plt.title(plt_title)
    plt.legend()
    plt.show()


def plotMap_same_scale(x_grnd, x_pred, plot_title=None, vmax_vmin='fixed', v_min=0.0, v_max=0.5,
    sv_plt=False, sv_path=None, sv_idx=None, custom_cmap=None, show_plot= None, lk_bck_days= 7):
    """ function: plots both the ground-truth and predicted results on similar scale 
    vmin and vmax are static, and pre-defined
    x_grnd: ground-truth data 
    x_pred: predicted data 
    plot_title: optional overall caption for the two plots
    vmax_vmin: value is 'fixed' or automatically computed; flag for determining, if vmax and vmin for plots are fixed or dynamically computed
    sv_plt: whether to save plotted figure 
    sv_path: path for saving plots
    sv_idx: index used for saving plotted figure
    custom_cmap: customized cmap for plotting
    show_plot: flag for showing plot on the fly """

    fig, axs = plt.subplots(nrows=1, ncols=2, dpi=300)
    # find minimum of minima & maximum of maxima
    if vmax_vmin== 'fixed':
        v_min_plt, v_max_plt = v_min, v_max
    else:       
        v_min_plt = np.min([np.min(x_grnd), np.min(x_pred)])
        v_max_plt = np.max([np.max(x_grnd), np.max(x_pred)])

    cmap_nw = plt.get_cmap('RdBu').copy()
    cmap_nw.set_bad(color='orange')

    if custom_cmap ==None:
        custom_cmap = plt.get_cmap('RdBu').copy()
    
    custom_cmap2 = custom_cmap.copy()
    custom_cmap.set_bad(color='white')
    # custom_cmap.set_under(color='magenta', alpha= 0)

    im1 = axs[0].imshow(x_grnd, vmin=v_min_plt, vmax=v_max_plt, aspect='equal', cmap=custom_cmap)
    axs[0].set_title('Ground-truth SMAP')
    axs[0].set_ylabel('y-pixel')
    axs[0].set_xlabel('x-pixel')
    
    im2 = axs[1].imshow(x_pred, vmin=v_min_plt, vmax=v_max_plt, aspect='equal', cmap=custom_cmap2)
    axs[1].set_title('Gap-filled SMAP; ' + 'look-back days= ' + str(lk_bck_days))
    # axs[1].set_title('Gap-filled SMAP; look-back= 5 days')
    axs[1].set_ylabel('y-pixel')
    axs[1].set_xlabel('x-pixel')

    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([1.0, 0.2, 0.02, 0.5])
    fig.colorbar(im2, cax=cbar_ax)

    if plot_title !=None:
        fig.suptitle(plot_title, fontsize=16)

    fig.tight_layout()
    if sv_plt== True:
        # plt.imsave(sv_path + 'gap-filling result for day' + ' ' + str(sv_idx)  + '.png', fig)
        plt.savefig(sv_path + 'result for' + ' ' + 
        str(sv_idx)  + '.png', bbox_inches='tight') 
    
    if show_plot== True or show_plot== None:
        plt.show()
    else:
        plt.close(fig)



def plotMap_SmapCygnss_same_scale(x_grnd_smap, x_grnd_cygnss, x_pred_smap, x_pred_cygnss,
    x_pred_comb, x_pred_comb_filtd, plot_title=None, vmax_vmin='fixed', 
    v_min=0.0, v_max=0.5, sv_plt=False, sv_path=None, sv_idx=None, 
    custom_cmap=None, show_plot= None, lk_bck_days= 7):
    """ function: plots both the ground-truth and predicted results on similar scale 
    vmin and vmax are static, and pre-defined
    x_grnd: ground-truth data 
    x_pred: predicted data 
    plot_title: optional overall caption for the two plots
    vmax_vmin: value is 'fixed' or automatically computed; flag for determining, if vmax and vmin for plots are fixed or dynamically computed
    sv_plt: whether to save plotted figure 
    sv_path: path for saving plots
    sv_idx: index used for saving plotted figure
    custom_cmap: customized cmap for plotting
    show_plot: flag for showing plot on the fly """


    txt_fnt = 16
    lb_sz = 14
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15,15), dpi=300)

    # find minimum of minima & maximum of maxima
    if vmax_vmin== 'fixed':
        v_min_plt, v_max_plt = v_min, v_max
    else:       
        v_min_plt = np.min([np.min(x_grnd_smap), np.min(x_grnd_cygnss), np.min(x_pred_smap), np.min(x_pred_comb)])
        v_max_plt = np.max([np.max(x_grnd_smap), np.min(x_grnd_cygnss), np.min(x_pred_smap), np.min(x_pred_comb)])

    cmap_nw = plt.get_cmap('RdBu').copy()
    cmap_nw.set_bad(color='orange')

    if custom_cmap ==None:
        custom_cmap = plt.get_cmap('RdBu').copy()
    
    custom_cmap2 = custom_cmap.copy()
    custom_cmap.set_bad(color='white')
    # custom_cmap.set_under(color='magenta', alpha= 0)

    im1 = axs[0, 0].imshow(x_grnd_smap, vmin=v_min_plt, vmax=v_max_plt, aspect='equal', cmap=custom_cmap)
    axs[0, 0].set_title('Ground-truth SMAP', fontsize = txt_fnt)
    axs[0, 0].set_ylabel('y-pixel', fontsize = txt_fnt)
    axs[0, 0].set_xlabel('x-pixel', fontsize = txt_fnt)
    axs[0, 0].tick_params(axis='both', labelsize=lb_sz)

    im2 = axs[0, 1].imshow(x_grnd_cygnss, vmin=v_min_plt, vmax=v_max_plt, aspect='equal', cmap=custom_cmap)
    axs[0, 1].set_title('Ground-truth CYGNSS', fontsize = txt_fnt)
    axs[0, 1].set_ylabel('y-pixel', fontsize = txt_fnt)
    axs[0, 1].set_xlabel('x-pixel', fontsize = txt_fnt)
    axs[0, 1].tick_params(axis='both', labelsize=lb_sz)
    
    im3 = axs[1, 0].imshow(x_pred_smap, vmin=v_min_plt, vmax=v_max_plt, aspect='equal', cmap=custom_cmap2)
    axs[1, 0].set_title('Gap-filled SMAP; ' + 'look-back measurements= ' + str(lk_bck_days), fontsize = txt_fnt)
    # axs[1].set_title('Gap-filled SMAP; look-back= 5 days')
    axs[1, 0].set_ylabel('y-pixel', fontsize = txt_fnt)
    axs[1, 0].set_xlabel('x-pixel', fontsize = txt_fnt)
    axs[1, 0].tick_params(axis='both', labelsize=lb_sz)

    im4 = axs[1, 1].imshow(x_pred_cygnss, vmin=v_min_plt, vmax=v_max_plt, aspect='equal', cmap=custom_cmap)
    axs[1, 1].set_title('Gap-filled CYGNSS; ' + 'look-back measurements= ' + str(lk_bck_days), fontsize = txt_fnt)
    axs[1, 1].set_ylabel('y-pixel', fontsize = txt_fnt)
    axs[1, 1].set_xlabel('x-pixel', fontsize = txt_fnt)
    axs[1, 1].tick_params(axis='both', labelsize=lb_sz)

    im5 = axs[2, 0].imshow(x_pred_comb, vmin=v_min_plt, vmax=v_max_plt, aspect='equal', cmap=custom_cmap2)
    axs[2, 0].set_title('Gap-filled SMAP+CYGNSS; ' + 'look-back measurements= ' + str(lk_bck_days), fontsize = 16)
    # axs[1].set_title('Gap-filled SMAP; look-back= 5 days')
    axs[2, 0].set_ylabel('y-pixel', fontsize = txt_fnt)
    axs[2, 0].set_xlabel('x-pixel', fontsize = txt_fnt)
    axs[2, 0].tick_params(axis='both', labelsize=lb_sz)

    im6 = axs[2, 1].imshow(x_pred_comb_filtd, vmin=v_min_plt, vmax=v_max_plt, aspect='equal', cmap=custom_cmap)
    axs[2, 1].set_title('Gap-filled SMAP+CYGNSS+filtered; ' + 'look-back measurements= ' + str(lk_bck_days), fontsize = 16)
    axs[2, 1].set_ylabel('y-pixel', fontsize = txt_fnt)
    axs[2, 1].set_xlabel('x-pixel', fontsize = txt_fnt)
    axs[2, 1].tick_params(axis='both', labelsize=lb_sz)

    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([1.0, 0.2, 0.02, 0.5])
    fig.colorbar(im6, cax=cbar_ax)

    if plot_title !=None:
        fig.suptitle(plot_title, fontsize=16)

    fig.tight_layout()
    if sv_plt== True:
        # plt.imsave(sv_path + 'gap-filling result for day' + ' ' + str(sv_idx)  + '.png', fig)
        plt.savefig(sv_path + 'result for' + ' ' + 
        str(sv_idx)  + '.png', bbox_inches='tight') 
    
    if show_plot== True or show_plot== None:
        plt.show(fig)
    else:
        plt.close(fig)




def plot_train_hist(train_loss, val_loss, EPOCHS, plot_title=None, y_label=None, 
    x_label=None, name_for_save=None, sv_path=None):
    """ function: plots model training and validation loss
    train_loss: training loss to be plotted
    val_loss: validation loss to be plotted 
    Epochs: number of training epochs
    plot_title: plot title 
    y_label: y-axis label
    x_label: x-axis label 
    name_for_save: name used for saving plot
    sv_path: directory for saving plot """

    plt.figure(dpi=300)
    x = np.arange(1, EPOCHS+1)
    plt.plot(x, train_loss, label = "training loss")
    plt.plot(x, val_loss, label = "validation loss")
    plt.xlabel('Epochs')
    if y_label !=None:
        plt.ylabel(y_label)
    if x_label != None:
        plt.xlabel(x_label)
    if plot_title !=None:
        plt.title(plot_title)
    plt.legend(loc='upper right')
    if name_for_save != None:
        if sv_path !=None:
            plt.savefig(sv_path + name_for_save)    #save in specified directory
        else:
            plt.savefig(name_for_save)      #save in current directory
    plt.show()

