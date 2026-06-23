### auxiliary functions for preprocessing features for input to the NeoGx Machine Learning classifier
import numpy as np
import re
from collections import Counter
from pyhpo import HPOTerm 
## deliberately written this way to avoid calling a pyhpo.OntologyClass instance
## functions in this module should be called by a script that has already loaded the ontology; will enable batch processing

## functions for basic features: sex, level IV age, and gestational age at birth

def encode_sex(sex: str) -> dict:
    """
    one-hot encodes sex as a binary feature (male/not male)
    
    Args:
    sex (string)

    Returns:
    dict: a dictionary with key "sex__male" and 
    """
    sex_dict = dict()
    sex_dict['sex__male'] = 1*(sex.lower() == 'male')
    return sex_dict

def encode_level4_age(age:float)->dict:
    """
    args
    age (float): days between birth and level IV NICU admission

    returns:
    dict keyed with feature name and with value equal to integer number of days between birth and level IV admission
    """
    return {'level4_age_days':int(age)}

def encode_gestational_age(ga_weeks: float) -> dict:
    """
    Returns several encodings of gestational age keyed by feature names

    Args:
    ga_weeks (float): The gestational age in weeks.

    Returns:
    dict: with raw ga_weeks value (ga_weeks), smallest integer below (ga_weeks__complete), or lower endpoint of GA bin (ga_weeks__ord)
    """
    for endpt in [42,37,35,32,28,0]:
        if ga_weeks >= endpt:
            return {
                'ga_weeks':ga_weeks,
                'ga_weeks__complete':np.floor(ga_weeks),
                'ga_weeks__ord':endpt,
                }

## functions for birth weight Z score: lookup LMS parameters, linearly interpolate between complete GA endpoints, calculate Z score
import json
BIRTH_CHART_FILE = '../assets/olsen_2010.json'
with open(BIRTH_CHART_FILE,'r') as f:
    BIRTH_CHART = json.load(f)
BIRTH_CHART_NAME = BIRTH_CHART_FILE.split('/')[-1].split('.json')[0]

def measure_lms_to_z(birth_weight_kgs:float,L:float,M:float,S:float)->float:
    """args:
    birth_weight_kgs (float): birth weight measured in kilograms
    L (float): lambda; skew parameter for the distribution
    M (float): median value for distribution
    S (float): SD/coefficient of variation

    returns birth weight Z score adjusted for sex and gestational age
    """
    if L == 0:
        z_score = (birth_weight_kgs - M) / S
    else:
        z_score = ((birth_weight_kgs / M)**L - 1) / (L * S)
    return z_score

def interpolate_param(sex:str, ga_wks:float, param_key:str,birth_chart:dict=BIRTH_CHART)->float:
    """args
    sex (str): individual's sex ("m" or "f")
    ga_wks (float): gestational age measured in weeks
    param_key (str): name of parameter to be interpolated ("L", "M", or "S")
    chart (dict): json-like nested dictionary keyed by <sex> ("m"|"f"): <parameter> ("L"|"M"|"S"): <parameter value>

    returns the linearly interpolated value of the parameter (birth chart only has values for integer values of ga_wks)
    """
    ga_lower = np.floor(ga_wks)
    ga_upper = np.ceil(ga_wks)
    param_lower = birth_chart[sex][str(int(ga_lower))][param_key]
    param_upper = birth_chart[sex][str(int(ga_upper))][param_key]
    if ga_lower==ga_upper:
        return param_lower
    else:
        m = (param_upper-param_lower)/(ga_upper-ga_lower)
        return param_lower + m*(ga_wks - ga_lower)

def interpolate_lms(sex:str,ga_wks:float, birth_chart:dict=BIRTH_CHART)->dict[str,float]:
    """Calculates interpreted LMS parameters
    
    args
    sex (str): individual's sex ("m" or "f")
    ga_wks (float): gestational age measured in weeks
    chart (dict): json-like nested dictionary keyed by <sex> ("m"|"f"): <parameter> ("L"|"M"|"S"): <parameter value>

    returns dict with L, M, and S values
    """
    L = interpolate_param(sex,ga_wks,'L',birth_chart)
    M = interpolate_param(sex,ga_wks,'M',birth_chart)
    S = interpolate_param(sex,ga_wks,'S',birth_chart)

    return dict(L=L,M=M,S=S)

def calculate_bw_z_score(sex:str, ga_wks:float, bw_kgs:float, birth_chart:dict=BIRTH_CHART)->float:
    """Calculates birth weight Z-score by LMS method, linearly interpolating LMS parameters
    args:
    sex (str): individual's sex (will be normalized to "m" or "f")
    ga_wks (float): gestational age measured in weeks
    birth_weight_kgs (float): birth weight measured in kilograms
    chart (dict): json-like nested dictionary keyed by <sex> ("m"|"f"): <parameter> ("L"|"M"|"S"): <parameter value>

    returns (float) sex- and gestational-age-adjusted birth weight Z score
    """
    sex = sex.lower()[0] # normalize sex to match birth chart lookup keys
    if sex not in 'mf':
        #raise ValueError("Sex must be 'Male/Female', 'male/female', 'M/F', 'm/f', ...")
        return np.nan
    if np.isnan(ga_wks):
        #raise ValueError("Gestational age cannot be null")
        return np.nan
    if np.isnan(bw_kgs):
        #raise ValueError("Birthweight cannot be null")
        return np.nan
    
    ga_wks = ga_wks - 3/7 # saved chart was offset to integer weeks to avoid incurring rounding errors when saving; compensate for this here
    if ga_wks < 23: # force ga within range for which LMS parameters are defined/interpolable
        ga_wks = 23
    elif ga_wks > 41:
        ga_wks = 41
    lms = interpolate_lms(sex, ga_wks, birth_chart)
    z_score = measure_lms_to_z(bw_kgs, **lms)
    return z_score

def encode_birth_weight(sex:str, gest_age_wks:float, birth_weight_kgs:float, chart:dict=BIRTH_CHART, chart_name:str='olsen_2010')->dict[str,float]:
    """args
    sex (str): sex of the individual
    gest_age_weeks (float): gestational age, measured in weeks (possibly with decimal part for days)
    birth_weight_kgs (float): birth weight in kilograms
    chart (dict): json-like nested dictionary keyed by <sex> ("m"|"f"): <parameter> ("L"|"M"|"S"): <parameter value>
    chart_name (str): name of the birth chart used, for feature name. Default and current only option is olsen_2010

    returns:
    dict keyed by different feature encodings and values of feature encodings 
    (raw birth weight, Z-score, and "scooped" Z-scores zeroing out scores with small absolute value)
    """
    z = calculate_bw_z_score(sex, gest_age_wks, birth_weight_kgs, birth_chart=chart)

    return {
        'birth_weight_kgs':birth_weight_kgs,
        f"birth_weight_{chart_name}_z":z,
        f"birth_weight_{chart_name}_z__abs_ge_1":z if np.abs(z)>=1 else 0,
        f"birth_weight_{chart_name}_z__abs_ge_2":z if np.abs(z)>=2 else 0,
        f"birth_weight_{chart_name}_z__abs_ge_3":z if np.abs(z)>=3 else 0,
        }

# functions for HPO ontology navigation and information content-based feature calculation

def calculate_descendants_of_term(term:HPOTerm)->set[HPOTerm]:
    """
    args:
        term (pyhpo.HPOTerm): an HPO term object
    returns:
        the set of descendant terms in HPO graph (including the input term)
    """
    desc = {term}
    for child in term.children:
        desc = desc | calculate_descendants_of_term(child)
    return desc

def calculate_descendants_of_set(X:set)->set[HPOTerm]:
    """
    args:
        term (set[pyhpo.HPOTerm]): a set of HPO term objects
    returns:
        the set of descendant terms in HPO graph (including the input set)
    """
    desc = set()
    for x in X:
        desc = desc | calculate_descendants_of_term(x)
    return desc

def calculate_ancestors_of_set(X:set)->set[HPOTerm]:
    """
    args:
        term (set[pyhpo.HPOTerm]): a set of HPO term objects
    returns:
        the set of ancestor terms in HPO graph (including the input set)
    """
    anc = set(X)
    for x in X:
        anc = anc | x.all_parents
    return anc

def calculate_ic_of_set(X:set[HPOTerm],ic_map)->set[HPOTerm]:
    """
    args:
        X: input set of HPO Terms
        (requires that pyhpo.OntologyClass object has been initialized with 'ic_phrank__marginal')
    returns:
        phrank IC of input set
    """
    return sum(
        #x.information_content.custom['ic_phrank__marginal'] 
        ic_map[x.id] 
        for x in calculate_ancestors_of_set(X)
    )

def calculate_pheno_ic_feature(
    subject_pheno_set:set[HPOTerm], 
    ic_map:dict,
    feature_term:HPOTerm, 
    feature_match:set[HPOTerm]=None, 
    cone:str='lower',
    ) -> float:
    """
    Calculates phenotype information content (IC) 
    for a given set of patient phenotype terms 
    relative to a chosen phenotype term

    Args:
    feature_term (str): HPO term ID for the chosen feature term
    subject_pheno_set (set): set of pyhpo.HPOTerms associated with the given subject
    feature_match (set): set of pyhpo.HPOTerms related to feature_term 
        (default None, consequently use descendant terms of feature_term)
    cone (str): method for calculating feature_match set if not provided; 
        defaults to "lower", meaning only IC for terms below the feature is measured

    Returns:
    float: the information content value for the chosen feature_term associated to the input patient_pheno_set
    """
    if not feature_match:
        if (cone == 'full') or (cone == 'bidirectional'):
            feature_match = {feature_term} | calculate_descendants_of_term(feature_term) | feature_term.all_parents ## note pyhpo.HPOTerm.all_parents does not return the original input term
        else:
            feature_match = calculate_descendants_of_term(feature_term)
    feature_match = set(feature_match)
    return sum(ic_map[x.id] for x in calculate_ancestors_of_set(feature_match & subject_pheno_set))

def make_hpo_pheno_vector(
        feature_set:set[HPOTerm], 
        subject_pheno_set:set[HPOTerm],
        ic_map:dict[str,float], 
        feature_match_map:dict=None, 
        cone:str='lower', 
        suffix:str=None) -> dict:
    """
    For a chosen set of HPO feature terms, calculates the phenotype encoding of each. 
    Returns all values in a dictionary keyed by f"{term_id}__{encoding}"

    Args:
    feature_set (set): chosen set of HPO feature terms to be provided to ML model
    patient_pheno_set (set): set of HPO terms associated to subject
    ic_map (dict): mappiong of HPO IDs to marginal Phrank IC score
    feature_match_map (dict): dict keyed by feature terms with values equal to intended feature_match for each feature_term
    cone (str): method for calculating associated terms to each member of feature_set
    name (str): name of feature encoding method; will automatically be named using "cone" arg

    Returns:
    dict: keys are f"{hpo_id}__{encoding_type}", values are corresponding IC values
    """
    if not suffix is None:
        pass
    else:
        if feature_match_map:
            suffix = ''
        else:
            if cone in ['full','bidirectional']:
                feature_match_map = {f: calculate_descendants_of_term(f)|f.all_parents for f in feature_set}
                suffix = '__bidirectional'
            else:
                feature_match_map = {f: calculate_descendants_of_term(f) for f in feature_set}
                suffix = '__lower'

    feature_val_dict = dict()
    for f,f_match in feature_match_map.items():
        feature_val = calculate_pheno_ic_feature(
            subject_pheno_set=subject_pheno_set,
            feature_term=f,
            feature_match=f_match,
            cone=cone,
            ic_map=ic_map,
            )
        feature_val_dict[f"{f.id}{suffix}"] = feature_val 
    return feature_val_dict

def classify_infec_test(name:str)->str:
    '''function for classifying an individual lab test (based on name) into one of our categories of interest (CULTURE/PCR/OTHER)
    args:
    name (str): name of lab test (from "DESCRIPTION" field)
    
    returns (str): classification of the lab test name as CULTURE, PCR, or OTHER
    '''
    if re.search('MRSA',name, flags=re.IGNORECASE):
        return 'ignore-MRSA'
    if re.search('wound',name, flags=re.IGNORECASE) and re.search('culture',name, flags=re.IGNORECASE):
        return 'ignore-wound'
    if re.search('lower resp',name, flags=re.IGNORECASE):
        return 'ignore-lower respiratory'
    if re.search('culture',name, flags=re.IGNORECASE):
        return 'BLOOD_CULTURE'
    if re.search('PCR',name, flags=re.IGNORECASE) or (
        re.search('respir',name, flags=re.IGNORECASE) 
        and re.search('infect',name, flags=re.IGNORECASE) 
        and re.search('array',name, flags=re.IGNORECASE)
    ):
        return 'PCR_VIRAL'
    if (name is np.nan) or (name is None) or (bool(name) is False):
        return 'ignore-null'
    return 'OTHER_INFECTION'

def make_infec_test_counter(lab_name_list:list[str], encoding:str='indicator', suffix:str=None)->dict[str,int]:
    '''function to generate infectious disease features from a list of lab test names.
    suffix can be provided to indicate if the list of labs was subject to any filters (e.g. all orders, abnormal results only, time constraint)

    args:
    lab_name_list (list): the names of the lab tests
    suffix (str): suffix to add to the feature keys

    returns: dictionary keyed by classifications of lab tests (with any user-supplied suffixes)
    '''
    clf_counter = Counter(map(classify_infec_test,lab_name_list))
    result = dict()
    encoding_key = 'indicator'
    if encoding=='count':
        encoding_key = 'count'
    else: 
        encoding_key = 'indicator'

    for key in ['BLOOD_CULTURE','PCR_VIRAL','OTHER_INFECTION']:
        result['__'.join([key,suffix,encoding_key])] = clf_counter[key]
    if encoding_key=='indicator':
        result = {k:1*(v>0) for k,v in result.items()} ### change counts to binary indications
    return result

def infec_order_result_to_categorical(lab_name_result_list:list[tuple[str,bool]])->int:
    if lab_name_result_list != lab_name_result_list:
        return 0
    classified_labs_with_results = [
        (classify_infec_test(name),res) 
        for name,res in lab_name_result_list 
        if 'ignore' not in classify_infec_test(name)
        ]
    if len(classified_labs_with_results)==0:
        return 0
    if any([(lab_res[-1]==True) for lab_res in classified_labs_with_results]):
        return 2
    else:
        return 1
    
