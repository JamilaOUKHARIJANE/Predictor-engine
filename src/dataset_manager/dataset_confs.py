import os

case_id_col = {}
activity_col = {}
resource_col = {}
timestamp_col = {}
label_col = {}
pos_label = {}
neg_label = {}
dynamic_cat_cols = {}
dynamic_activity_col = {}
static_cat_cols = {}
dynamic_num_cols = {}
static_num_cols = {}
filename = {}

logs_dir = ""

#### Flood management log settings ####
dataset = "fmplog"

filename[dataset] = os.path.join(logs_dir, "FMPlog.csv")

case_id_col[dataset] = "Case ID"
activity_col[dataset] = "Activity"
timestamp_col[dataset] = "Complete Timestamp"
label_col[dataset] = "label"
neg_label[dataset] = "noAdapt"
pos_label[dataset] = "Adapt"

# features for classifier
dynamic_activity_col[dataset] = ["Activity"]
static_cat_cols[dataset] = ["Materiel Ressource","Human Ressource"]
static_num_cols[dataset] = []
dynamic_cat_cols[dataset] = ["risque level","Activity"]
dynamic_num_cols[dataset] = ["water flow","water level"]

#### Traffic fines settings ####

#### Sepsis Cases settings ####
dataset= "sepsis_cases"
filename[dataset] = os.path.join(logs_dir, "sepsis_cases.xes")

case_id_col[dataset] = "case:concept:name"
activity_col[dataset] = "concept:name"
resource_col[dataset] = "org:group"
timestamp_col[dataset] = "time:timestamp"
label_col[dataset] = "label"
neg_label[dataset] = "noAdapt"
pos_label[dataset] = "Adapt"

# features for classifier
dynamic_activity_col[dataset] = ["concept:name"]
dynamic_cat_cols[dataset] = ["concept:name"] # i.e. event attributes
static_cat_cols[dataset] = ['Diagnose'] # i.e. case attributes that are known from the start
dynamic_num_cols[dataset] = ['CRP', 'LacticAcid', 'Leucocytes']
static_num_cols[dataset] = ['Age','DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG',
                       'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
                       'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture',
                       'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg',
                       'Hypotensie',    'SIRSCritHeartRate', 'SIRSCritTachypnea',
                       'SIRSCritTemperature']



dataset = "hospital_billing"

filename[dataset] = os.path.join(logs_dir, "hospital_billing.xes")

case_id_col[dataset] = "case:concept:name"
activity_col[dataset] = "concept:name"
resource_col[dataset] = "org:resource"
timestamp_col[dataset] = "time:timestamp"
label_col[dataset] = "label"
neg_label[dataset] = "noAdapt"
pos_label[dataset] = "Adapt"

# features for classifier
dynamic_activity_col[dataset] = ["concept:name"]
dynamic_cat_cols[dataset] = ["concept:name",'actOrange', 'actRed', 'blocked', 'caseType', 'diagnosis','flagA','flagB', 'flagC',
                             'flagD', 'msgCode', 'msgType', 'state',
                             'version', 'isCancelled', 'isClosed', 'closeCode']
static_cat_cols[dataset] = ['speciality']
dynamic_num_cols[dataset] = ['msgCount']
static_num_cols[dataset] = []


