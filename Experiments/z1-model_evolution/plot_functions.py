import os, pickle
import numpy as np
import pandas as pd

from matplotlib.ticker import MaxNLocator


sampleSkip = 30
start_index, end_index = 50, 350

y_label_size = 20
y_label_tick_size = 20
text_size = 20

normal_color = 'teal'
anomalous_color = 'red'

cluster_id_marker = {
    0:'o',
    1:'>',
    2:'x',
    3:'<'
}
cluster_id_linestyle = {
    1:'-',
    2:'-.',
    3:'--'        
}

cluster_id_color = {
    0: '#1f77b4',
    1: '#ff7f0e',
    2: 'green',
    3: 'orange'
}

cluster_id_color_ODS = {
    1: '#1f77b4',
    2: '#ff7f0e',
    3: 'green',
    4: 'orange'
}

cluster_id_alpha_wdbscan = {
    0: 1,
    1: 0.4,
    2: 0.2
}

def load_pickles(method='ODS'):
    pickles = {}
    directory = method+'_variables/'
    for element in os.listdir(directory):
        if 'pickle' in element:
            var_name = element.split('.')[0].split(method+'_')[1]

            with open(directory+element, 'rb') as f:
                pickles[var_name] = pickle.load(f)
                
    return pickles

def load_data():
    ods_data = load_pickles(method='ODS')
    wdbscan_data = load_pickles(method='wDBScan')
    
    return ods_data, wdbscan_data

def plot_normal_vs_outlier_cluster_evolution(ax=None, data=None, method='ODS'):
    
    if method == 'ODS':
    
        # ax.plot(np.array(list(range(start_index, end_index))),
        #     data['core_clusters_number'][start_index:end_index],
        #         color=cluster_id_color[0],
        #         linestyle='-',
        #         linewidth=3,
        #         label='normal')

        for cluster_id in data['clusters_radius']:
            radius_data = pd.DataFrame(data['clusters_radius'][cluster_id]['radius'], index=data['clusters_radius'][cluster_id]['index'], columns=['radius'])
            normal_values = pd.Series(data['core_clusters_number'])
            indexes = radius_data.index

            ax.plot(indexes[(indexes<end_index) & (indexes>50)],
                    normal_values[indexes[(indexes<end_index) & (indexes>50)]],
                    c=cluster_id_color_ODS[cluster_id],
                    linewidth=3)

        ax.plot(np.array(list(range(start_index, end_index))),
                data['outrlier_clusters_number'][start_index:end_index],
                color=anomalous_color,
                linestyle='-.',
                linewidth=3,
                label='anomalous')
        

    elif method == 'wDBScan':
        

        normal_values = pd.Series(data['normal_clusters'][:300])
        index = (normal_values[normal_values==2].index)

        ax.plot(np.array(range(len(data['normal_clusters'][:300])))+50,
                data['normal_clusters'][:300],
                color=cluster_id_color[0],
                linestyle='-',
                linewidth=3,
                label='normal c1'
        )
        ax.plot(index[index<200]+50,
                normal_values.loc[index[index<200]],
                '--', 
                linewidth=3,
                c=cluster_id_color[1],
                label='normal c2')
        ax.plot(index[index>200]+50,
                normal_values.loc[index[index>200]],
                '--', 
                linewidth=3,
                c=cluster_id_color[1])        

        ax.plot(np.array(range(len(data['anomalous_clusters'][:300])))+50,
                data['anomalous_clusters'][:300],
                color=anomalous_color,
                linestyle='-.',
                linewidth=3,
                label='anomalous'
        )


        ax.set_ylabel('#cluster', fontsize=y_label_size, labelpad=20)
        ax.legend()
        ax.set_zorder(1)

    else:
        pass
    
    ax.tick_params(axis='both', which='major', labelsize=y_label_tick_size)
    ax.tick_params(axis='both', which='minor', labelsize=y_label_tick_size)
    
def plot_assigned_labels(ax=None, data=None, method='ODS'):
    
    if method == 'ODS':
    
        is_outlier_ = pd.DataFrame(data['is_outlier'])
        is_outlier_['id'] = data['sample_cluster_id']

        is_outlier_=is_outlier_[start_index:end_index]

        normal_index = list(is_outlier_[is_outlier_[0]==False].index)
        normal_value = list(is_outlier_.loc[normal_index]['id'])

        normal_value_1 = is_outlier_.loc[normal_index]['id']
        normal_value_1 = normal_value_1[normal_value_1==1]

        normal_value_2 = is_outlier_.loc[normal_index]['id']
        normal_value_2 = normal_value_2[normal_value_2==2]

        anomalous_index = list(is_outlier_[is_outlier_[0]==True].index)
        anomalous_value = list(is_outlier_.loc[anomalous_index]['id'])

        # ax.scatter(normal_index,
        #            normal_value,
        #            color=normal_color,
        #            s=4
        # )

        ax.scatter(normal_value_1.index,
                   normal_value_1,
                   color=cluster_id_color_ODS[1],
                   s=5
        )        

        ax.scatter(normal_value_2.index,
                   normal_value_2,
                   color=cluster_id_color_ODS[2],
                   s=5
        )        


        ax.scatter(anomalous_index,
                   anomalous_value,
                   color=anomalous_color,
                   s=5
        )
                
    elif method == 'wDBScan':
        
        local_labels = pd.DataFrame(data['output_labels'][:300])

        # ax.scatter(local_labels[local_labels[0]>=0].index+50,
        #            local_labels[local_labels[0]>=0][0]+1,
        #            color=normal_color,
        #            s=4
        # )

        # ax.scatter(local_labels[local_labels[0]<0].index+50,
        #            np.ones(len(local_labels[local_labels[0]<0].index+50))*5,
        #            color=anomalous_color,
        #            s=4
        # )

        ax.scatter(local_labels[local_labels[0]==0].index+50,
                   local_labels[local_labels[0]==0]+1,
                   s=5,
                   color=cluster_id_color[0],
                   marker=cluster_id_marker[0])
        ax.scatter(local_labels[local_labels[0]==1].index+50,
                   local_labels[local_labels[0]==1]+1,
                   s=5,
                   color=cluster_id_color[1],
                   marker=cluster_id_marker[0])        
        ax.scatter(local_labels[local_labels[0]<0].index+50,
                   local_labels[local_labels[0]<0],
                   s=5,
                   color=anomalous_color,
                   marker='x')  

        ax.set_ylabel('labels', fontsize=y_label_size, labelpad=10)
    else:
        pass

    ax.tick_params(axis='both', which='major', labelsize=y_label_tick_size)
    ax.tick_params(axis='both', which='minor', labelsize=y_label_tick_size)    
    
    
def plot_radius(ax=None, data=None, method='ODS'):
    
    if method == 'ODS':

        for cluster_id in data['clusters_radius']:

            radius_data = pd.DataFrame(data['clusters_radius'][cluster_id]['radius'], index=data['clusters_radius'][cluster_id]['index'], columns=['radius'])

            ax.plot(radius_data.loc[start_index:end_index].index,
                    radius_data.loc[start_index:end_index,'radius'],
                    marker=cluster_id_marker[cluster_id],
                    linestyle=cluster_id_linestyle[cluster_id],
                    markersize=5)

        ax.plot(np.array(list(range(start_index, end_index))),
                data['updated_epsilons'][start_index+sampleSkip:end_index+sampleSkip])

        
        
    elif method == 'wDBScan':
        
        for index, element in enumerate(data['cluster_characteristics'][:300]):
            for clusters in element:
                ax.scatter(index+50, clusters[2], s=10, color=cluster_id_color[clusters[0]], marker=cluster_id_marker[clusters[0]])
        
        ax.set_ylabel('radius', fontsize=y_label_size, labelpad=5)                
    else:
        pass
           
    ax.yaxis.set_major_locator(MaxNLocator(3))        
    ax.tick_params(axis='both', which='major', labelsize=y_label_tick_size)
    ax.tick_params(axis='both', which='minor', labelsize=y_label_tick_size) 
    
def plot_first_center(ax=None, data=None, method='ODS'):
    
    if method == 'ODS':
    
        for cluster_id in data['clusters_radius']:

            ax.scatter(data['pca_centers'].loc[data['centers_clusterID_index'][cluster_id],0].loc[start_index:end_index].index,
                    data['pca_centers'].loc[data['centers_clusterID_index'][cluster_id],0].loc[start_index:end_index],
                    marker=cluster_id_marker[cluster_id],
                    linestyle=cluster_id_linestyle[cluster_id],
                    s=20
            )

        
        
    elif method == 'wDBScan':
        pca_centers = data['pca_centers']
        
        for cluster_id in pca_centers['labels'].unique():
            to_plot = pca_centers[pca_centers['labels']==cluster_id]
            ax.scatter(to_plot['indexes']+50, to_plot[0], s=10, color=cluster_id_color[cluster_id], marker=cluster_id_marker[cluster_id])
        ax.set_ylabel('$1^{st}$ c. center', fontsize=y_label_size)
        
    ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.tick_params(axis='both', which='major', labelsize=y_label_tick_size)
    ax.tick_params(axis='both', which='minor', labelsize=y_label_tick_size)     

def plot_second_center(ax=None, data=None, method='ODS'):
    
	if method == 'ODS':

		pca_centers = data['pca_centers']
		clusters_radius = data['clusters_radius']
		centers_clusterID_index = data['centers_clusterID_index']

		for cluster_id in clusters_radius:
			ax.scatter(pca_centers.loc[centers_clusterID_index[cluster_id],1].loc[start_index:end_index].index,
			        pca_centers.loc[centers_clusterID_index[cluster_id],1].loc[start_index:end_index],
			        marker=cluster_id_marker[cluster_id],
			        linestyle=cluster_id_linestyle[cluster_id],
			        s=20
			)

	elif method == 'wDBScan':
		
		pca_centers = data['pca_centers']

		for cluster_id in pca_centers['labels'].unique():
			to_plot = pca_centers[pca_centers['labels']==cluster_id]
			ax.scatter(to_plot['indexes']+50, to_plot[1], s=10, color=cluster_id_color[cluster_id], marker=cluster_id_marker[cluster_id]) 
		ax.set_ylabel('$2^{nd}$ c. center', fontsize=y_label_size)

	else:
		pass

	ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
	ax.tick_params(axis='both', which='major', labelsize=y_label_tick_size)
	ax.tick_params(axis='both', which='minor', labelsize=y_label_tick_size)		

def plot_weight(ax=None, data=None, method='ODS'):

	if method == 'ODS':
    
		clusters_radius = data['clusters_radius']
		weights = data['weights']

		for cluster_id in clusters_radius:
		    
			radius_data = pd.DataFrame(weights[cluster_id]['weight'], index=weights[cluster_id]['index'], columns=['weight'])

			ax.plot(radius_data.loc[start_index:end_index].index,
			        radius_data.loc[start_index:end_index,'weight'],
			        marker=cluster_id_marker[cluster_id],
			        linestyle=cluster_id_linestyle[cluster_id],
			        markersize=5)
		    
		    
		ax.axhline(y=data['promotion_threshold'], color='k', linestyle='--')
		# ax.tick_params(axis="y",direction="in", pad=-22) 

	elif method == 'wDBScan':
		
		cluster_characteristics=data['cluster_characteristics']

		for index, element in enumerate(cluster_characteristics[:300]):
			for clusters in element:
				ax.scatter(index+50, clusters[3], s=10, color=cluster_id_color[clusters[0]], marker=cluster_id_marker[clusters[0]])
		ax.set_ylabel('weight', fontsize=y_label_size, labelpad=10)

		# ax.axhline(y=3, color='k', linestyle='--')

	else:
		pass
	
	# ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
	ax.tick_params(axis='both', which='major', labelsize=y_label_tick_size)
	ax.tick_params(axis='both', which='minor', labelsize=y_label_tick_size)  	

def plot_vertical_event(axs, events):
	for ax in axs:
		for event_index in events:
			ax.axvline(x=event_index, linestyle='--', color='k')	