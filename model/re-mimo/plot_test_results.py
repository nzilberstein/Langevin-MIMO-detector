import pickle
import numpy as np
import matplotlib.pyplot as plt

mod = 'qam_16'
caption = 'QAM-16, '
NT_label =  r'$N_{tr}$' + '='
NR_label = r'$N_{r}$' + '=64'


with open('./final_results'+'/network_fullcorr_validtn_results.pickle', 'rb') as handle:
	network_validtn_results_dict = pickle.load(handle)

with open('./final_results' + '/oampnet_fullcorr_16_validtn_results.pickle', 'rb') as handle:
	oampnet_16_validtn_results_dict = pickle.load(handle)

with open('./final_results' + '/oampnet_fullcorr_32_validtn_results.pickle', 'rb') as handle:
	oampnet_32_validtn_results_dict = pickle.load(handle)

oampnet_validtn_results_dict = {**oampnet_16_validtn_results_dict, **oampnet_32_validtn_results_dict}

def process_data(validtn_result):
	validtn_result_keys = np.asarray(list(validtn_result.keys()))
	validtn_result_values = np.asarray(list(validtn_result.values()))
	valid_indices = [i for i in range(len(validtn_result_values)) if validtn_result_values[i]>=0.96]
	min_valid_index = min(valid_indices)
	valid_indices = np.arange(min_valid_index, min_valid_index+6)
	print(valid_indices)
	result_keys = np.asarray(validtn_result_keys[valid_indices])
	result_values = 1.0 - np.asarray(validtn_result_values[valid_indices])
	return result_keys, result_values

NR = 64
NT_list = list(network_validtn_results_dict.keys())
corr_list = np.asarray([0.60, 0.70])
markers_list= ['o','v']

for NT in NT_list:
	plt.figure()
	for i, rho in enumerate(corr_list):

		rho_label = '(' + r'$\rho$' + '=' + str(rho) + ')'

		if (rho == 0.70):
			if (NT==16):
				init_point = 1
			elif(NT==32):
				init_point = 2
		elif (rho == 0.60):
			if (NT==16):
				init_point = 0
			elif (NT==32):
				init_point = 1
		else:
			init_point = 0

		network_validtn_result = network_validtn_results_dict[NT][rho]
		network_result_keys = list(network_validtn_result.keys())
		network_result_keys = np.asarray(network_result_keys, dtype=np.int32)
		network_result_values = 1.0 - np.asarray([network_validtn_result.get(key) for key in network_result_keys])
		plt.plot(network_result_keys[init_point: init_point+6], network_result_values[init_point: init_point+6], label='RE-MIMO '+rho_label, marker=markers_list[i], ms=7, color='r', fillstyle='none')

		oampnet_validtn_result = oampnet_validtn_results_dict[NT][rho]
		oampnet_result_keys = list(oampnet_validtn_result.keys())
		oampnet_result_keys = np.asarray(oampnet_result_keys, dtype=np.int32)
		oampnet_result_values = 1.0 - np.asarray(list(oampnet_validtn_result.values()))
		plt.plot(oampnet_result_keys[init_point: init_point+6], oampnet_result_values[init_point: init_point+6], label='OAMPNet '+rho_label , marker=markers_list[i], ms=6, color='k', fillstyle='none', ls=':')


		plt.yscale('log')
		plt.grid(True, which='both')
		plt.legend(loc='lower left', fontsize=11)
		plt.xlabel('SNR(dB)', fontsize=13)
		plt.ylabel('SER', fontsize=13)
		plt.title(caption + NT_label+ str(NT) + ', ' + NR_label, fontsize=13)
		plt.tight_layout()
		plt.savefig('./final_results/' + '/' + mod + '_NT' + str(NT) + '_NR64_fullcorr.pdf')
plt.show()
