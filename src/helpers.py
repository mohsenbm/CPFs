import numpy as np
import pandas as pd

# Helper functions to save error data for PFs and CPFs as dataframe

def cpf1_and_cpf2_error_dataframe(
    nsteps,
    nticks,
    data_CPF1_errors,
    data_CPF2_errors,
    data_PF4_errors,
    description):
    """
    Build dataframe for errors of PF/CPF with orders 1 & 2. Error data for PF4 is also included.
    """
    # ---- description column ----
    # description column will be padded with NaN to match len(ticks)
    description = description + [np.nan] * (nticks - len(description))

    # ---- CPF1 ----
    y_PF1_step_error = [data_CPF1_errors[i][0] for i in range(nticks)]
    y_PF1_rsteps_error = list(nsteps * np.array(y_PF1_step_error))
    y_PF1_error      = [data_CPF1_errors[i][3] for i in range(nticks)]
    y_CPF1symp_error = [data_CPF1_errors[i][4] for i in range(nticks)]
    y_CPF1com_error  = [data_CPF1_errors[i][5] for i in range(nticks)]
    y_CPF1com_step_error = [data_CPF1_errors[i][2] for i in range(nticks)]
    y_CPF1com_rsteps_error = list(nsteps * np.array(y_CPF1com_step_error))

    # ---- CPF2 ----
    y_PF2_step_error = [data_CPF2_errors[i][0] for i in range(nticks)]
    y_PF2_rsteps_error = list(nsteps * np.array(y_PF2_step_error))
    y_PF2_error      = [data_CPF2_errors[i][3] for i in range(nticks)]
    y_CPF2symp_error = [data_CPF2_errors[i][4] for i in range(nticks)]
    y_CPF2com_error  = [data_CPF2_errors[i][5] for i in range(nticks)]
    y_CPF2com_step_error = [data_CPF2_errors[i][2] for i in range(nticks)]
    y_CPF2com_rsteps_error = list(nsteps * np.array(y_CPF2com_step_error))

    # ---- PF4 ----
    y_PF4step_error = [data_PF4_errors[i][0] for i in range(nticks)]
    y_PF4_rsteps_error = list(nsteps * np.array(y_PF4step_error))
    y_PF4_error = [data_PF4_errors[i][1] for i in range(nticks)]


    # ---- dataframe ----
    data = {
        'description': description,
        'y_PF1_rsteps_error': y_PF1_rsteps_error,
        'y_PF1_error': y_PF1_error,
        'y_CPF1symp_error': y_CPF1symp_error,
        'y_CPF1com_error': y_CPF1com_error,
        'y_CPF1com_rsteps_error': y_CPF1com_rsteps_error,
        'y_PF2_rsteps_error': y_PF2_rsteps_error,
        'y_PF2_error': y_PF2_error,
        'y_CPF2symp_error': y_CPF2symp_error,
        'y_CPF2com_error': y_CPF2com_error,
        'y_CPF2com_rsteps_error': y_CPF2com_rsteps_error,
        'y_PF4_rsteps_error': y_PF4_rsteps_error,
        'y_PF4_error': y_PF4_error
    }
    
    return pd.DataFrame(data)

def cpf4_and_cpf6_error_dataframe(
    nsteps,
    nticks,
    data_CPF4_sym_errors,
    data_CPF6_sym_errors,
    description):
    """
    Build dataframe for errors of PF/CPF with orders 4 & 6.
    """
    # ---- description column ----
    # description column will be padded with NaN to match len(ticks)
    description = description + [np.nan] * (nticks - len(description))

    # ---- PF4 ----
    y_PF4step_error = [data_CPF4_sym_errors[i][0] for i in range(nticks)]
    y_PF4_rsteps_error = list(nsteps * np.array(y_PF4step_error))
    y_PF4_errors = [data_CPF4_sym_errors[i][2] for i in range(nticks)]

    # ---- PF6 ----
    y_PF6step_error = [data_CPF6_sym_errors[i][0] for i in range(nticks)]
    y_PF6_rsteps_error = list(nsteps * np.array(y_PF6step_error))
    y_PF6_errors = [data_CPF6_sym_errors[i][2] for i in range(nticks)]

    # ---- CPF4 ----
    y_CPF4step_error = [data_CPF4_sym_errors[i][1] for i in range(nticks)]
    y_CPF4_rsteps_error = list(nsteps * np.array(y_CPF4step_error))
    y_CPF4_errors = [data_CPF4_sym_errors[i][3] for i in range(nticks)]

    # ---- CPF6 ----
    y_CPF6step_error = [data_CPF6_sym_errors[i][1] for i in range(nticks)]
    y_CPF6_rsteps_error = list(nsteps * np.array(y_CPF6step_error))
    y_CPF6_errors = [data_CPF6_sym_errors[i][3] for i in range(nticks)]

    # ---- dataframe ----
    data = {
        'description': description,
        'y_PF4_rsteps_error': y_PF4_rsteps_error,
        'y_PF4_errors': y_PF4_errors,
        'y_PF6_rsteps_error': y_PF6_rsteps_error,
        'y_PF6_errors': y_PF6_errors,
        'y_CPF4_rsteps_error': y_CPF4_rsteps_error,
        'y_CPF4_errors': y_CPF4_errors,
        'y_CPF6_rsteps_error': y_CPF6_rsteps_error,
        'y_CPF6_errors': y_CPF6_errors
    }

    return pd.DataFrame(data)

def cpf1_and_cpf2_error_fixed_step_dataframe(
    size_steps_list,
    data_CPF1_errors,
    data_CPF2_errors,
    data_PF4_errors,
    description):
    """
    Build dataframe for fixed-step errors of PF/CPF with orders 1 & 2. Error data for fixed-step PF4 is also included.
    """
    # ---- description column ----
    # description column will be padded with NaN to match size_steps_list
    description = description + [np.nan] * (size_steps_list - len(description))

    # ---- CPF1 ----
    y_PF1_error      = [data_CPF1_errors[i][3] for i in range(size_steps_list)]
    y_CPF1symp_error = [data_CPF1_errors[i][4] for i in range(size_steps_list)]
    y_CPF1com_error  = [data_CPF1_errors[i][5] for i in range(size_steps_list)]
    
    # ---- CPF2 ----
    y_PF2_error      = [data_CPF2_errors[i][3] for i in range(size_steps_list)]
    y_CPF2symp_error = [data_CPF2_errors[i][4] for i in range(size_steps_list)]
    y_CPF2com_error  = [data_CPF2_errors[i][5] for i in range(size_steps_list)]

    # ---- PF4 ----
    y_PF4_error = data_PF4_errors


    # ---- dataframe ----
    data = {
        'description': description,
        'y_PF1_error': y_PF1_error,
        'y_CPF1symp_error': y_CPF1symp_error,
        'y_CPF1com_error': y_CPF1com_error,
        'y_PF2_error': y_PF2_error,
        'y_CPF2symp_error': y_CPF2symp_error,
        'y_CPF2com_error': y_CPF2com_error,
        'y_PF4_error': y_PF4_error
    }
    
    return pd.DataFrame(data)

def cpf1_and_cpf2_variable_perturbation_error_dataframe(
    nsteps,
    nticks,
    data_CPF1_errors,
    data_CPF2_errors,
    description):
    """
    Build dataframe for errors of PF/CPF with orders 1 & 2. Error data for PF4 is also included.
    """
    # ---- description column ----
    # description column will be padded with NaN to match len(ticks)
    description = description + [np.nan] * (nticks - len(description))

    # ---- CPF1 ----
    y_PF1_step_error = [data_CPF1_errors[i][0] for i in range(nticks)]
    y_PF1_rsteps_error = list(nsteps * np.array(y_PF1_step_error))
    y_PF1_error      = [data_CPF1_errors[i][3] for i in range(nticks)]
    y_CPF1symp_error = [data_CPF1_errors[i][4] for i in range(nticks)]
    y_CPF1com_error  = [data_CPF1_errors[i][5] for i in range(nticks)]
    y_CPF1com_step_error = [data_CPF1_errors[i][2] for i in range(nticks)]
    y_CPF1com_rsteps_error = list(nsteps * np.array(y_CPF1com_step_error))

    # ---- CPF2 ----
    y_PF2_step_error = [data_CPF2_errors[i][0] for i in range(nticks)]
    y_PF2_rsteps_error = list(nsteps * np.array(y_PF2_step_error))
    y_PF2_error      = [data_CPF2_errors[i][3] for i in range(nticks)]
    y_CPF2symp_error = [data_CPF2_errors[i][4] for i in range(nticks)]
    y_CPF2com_error  = [data_CPF2_errors[i][5] for i in range(nticks)]
    y_CPF2com_step_error = [data_CPF2_errors[i][2] for i in range(nticks)]
    y_CPF2com_rsteps_error = list(nsteps * np.array(y_CPF2com_step_error))


    # ---- dataframe ----
    data = {
        'description': description,
        'y_PF1_rsteps_error': y_PF1_rsteps_error,
        'y_PF1_error': y_PF1_error,
        'y_CPF1symp_error': y_CPF1symp_error,
        'y_CPF1com_error': y_CPF1com_error,
        'y_CPF1com_rsteps_error': y_CPF1com_rsteps_error,
        'y_PF2_rsteps_error': y_PF2_rsteps_error,
        'y_PF2_error': y_PF2_error,
        'y_CPF2symp_error': y_CPF2symp_error,
        'y_CPF2com_error': y_CPF2com_error,
        'y_CPF2com_rsteps_error': y_CPF2com_rsteps_error,

    }
    
    return pd.DataFrame(data)