"""
This module provides basic methods for unit conversion and calculation of basic wind plant variables
"""

def convert_power_to_energy(power_col,sample_rate_min=10.0):
    """
    Compute energy [kWh] from power [kw] and return the data column

    Args:
        df(:obj:`pandas.DataFrame`): the existing data frame to append to
        col(:obj:`string`): Power column to use if not power_kw
        sample_rate_min(:obj:`float`): Sampling rate in minutes to use for conversion, if not ten minutes

    Returns:
        :obj:`pandas.Series`: Energy in kWh that matches the length of the input data frame 'df'

    """
    energy_kwh=power_col*sample_rate_min/60.0
    return energy_kwh

def compute_gross_energy(net_energy, avail_losses, curt_losses, avail_type = 'frac', curt_type = 'frac'):
    """
    This function computes gross energy for a wind plant or turbine by adding reported availability and
    curtailment losses to reported net energy. Account is made of whether availabilty or curtailment loss data
    is reported in energy ('energy') or fractional units ('frac'). If in energy units, this function assumes that net energy,
    availability loss, and curtailment loss are all reported in the same units
    
    Args:
        net energy (numpy array of Pandas series): reported net energy for wind plant or turbine 
        avail (numpy array of Pandas series): reported availability losses for wind plant or turbine 
        curt (numpy array of Pandas series): reported curtailment losses for wind plant or turbine 
        
    Return:
        gross (numpy array of Pandas series): calculated gross energy for wind plant or turbine 
    """
    
    if (avail_type == 'frac') & (curt_type=='frac'):
        gross = net_energy / (1 - avail_losses - curt_losses)
    elif (avail_type == 'frac') & (curt_type=='energy'):
        gross = net_energy / (1 - avail_losses) + curt_losses
    elif (avail_type == 'energy') & (curt_type=='frac'):
        gross = net_energy / (1 - curt_losses) + avail_losses
    elif (avail_type == 'energy') & (curt_type=='energy'):
        gross = net_energy + curt_losses + avail_losses
        
    if (len(gross[gross < 0]) > 0) | (len(gross[gross < net_energy]) > 0):
        raise Exception('Gross energy cannot be negative or less than net energy. Check your input values')    
    if (len(net_energy[net_energy < 0]) > 0) | (len(avail_losses[avail_losses < 0]) > 0) | (len(curt_losses[curt_losses < 0]) > 0):
        raise Exception('Cannot have negative input values. Check your data')    

    return gross