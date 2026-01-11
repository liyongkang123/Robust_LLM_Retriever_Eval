'''
Using R is more appropriate
2025-11-08 00:30:00 updated
I currently think that using R is more accurate, so the R analysis results of this file.
'''

import os
import platform

# Check the current operating system, if it is a Windows system, force the system R to be specified
system_name = platform.system().lower()
print(f"System detected: {system_name}")
if 'windows' in system_name:
    # Force the system R to be specified in Windows
    os.environ["R_HOME"] = r"C:\Program Files\R\R-4.5.2"
    print(f"✅ Detected Windows — using system R at: {os.environ['R_HOME']}")
else:
    # Linux / macOS default according to the system or conda configuration
    print(f"✅ Detected {system_name.capitalize()} — using default R configuration.")


import rpy2.robjects as robjects
# robjects.r('Sys.setlocale("LC_ALL", "C")')  # Or "en_US.UTF-8"
robjects.r('Sys.setlocale(category = "LC_CTYPE", locale = "en_US.UTF-8")') #1. Set the locale of R (to prevent character encoding issues)
# Set the print width of R, ensure that all columns can be displayed side by side
robjects.r('options(width = 200)')  # Adjust the width according to your data

import pandas as pd
from rpy2.robjects import pandas2ri, conversion
# Import R packages
from rpy2.robjects.packages import importr

lme4 = importr('lme4')  # Need to install in R install.packages("lme4")
lmerTest = importr('lmerTest') # Need to install in R install.packages("lmerTest")
# Add to your rpy2 code
emmeans = importr('emmeans')

# Global constants: define the reference level
BASE_MODEL = 'contriever'  # Reference model
BASE_QUERY_TYPE = 'FACTOID'  # Reference query_type
BASE_TASK_TYPE = 'Question Answering'  # Reference task_type
BASE_CORPUS_TYPE = 'Wikipedia'  # Reference corpus_type

model_order = ['bm25','contriever', 'bge_m3', 'qwen3', 'linq', 'gte', 'reasonir', 'diver', 'bge_reasoner']  
corpus_order = ["General Web", "Scientific Paper", "Wikipedia", "Online Community",  "Domain KB", ]
task_order =   ['Passage Retrieval','Bio-Medical Retrieval','Question Answering','Duplicate Question Retrieval','Entity Retrieval','Citation-Prediction', 'Fact Checking', \
 'Argument Retrieval','StackExchange Post Retrieval', 'Code Retrieval', 'Theorem Retrieval', ]
query_type_order = ["FACTOID", "INSTRUCTION", "REASON", "EVIDENCE-BASED", "COMPARISON", "EXPERIENCE", "DEBATE", "NOT-A-QUESTION"]

import sys
import os
import pingouin as pg
import statsmodels.formula.api as smf

def read_data():
    data_path ='data/df_long_all_model_scores.csv'
    data = pd.read_csv(data_path)
    data = data.rename(columns={'ndcg@10': 'ndcg_10'})
    # Remove the data where Model is bm25
    # data = data[data['Model'] != 'bm25'].reset_index(drop=True)
    return data

def analysis_model_with_query_type_mixedlm(r_data):
    ''' When performing LMM analysis, an automatic reference level (reference level) will be selected, and the coefficient of this level will not be displayed separately, but will be included in the intercept term.
    We can also specify the reference level of the fixed effect during analysis as which model and reference query_type
    '''
    # Analysis 1:
    # C(Model) * C(query_type) + (1 | dataset_name/query_id)
    print("Analysis 1:")
    
    # Register r_data to the R global environment, so that R code can access it
    robjects.globalenv['r_data'] = r_data

    # Recreate the factors, ensure that the reference level is cleared
    # First convert to a character type, then convert to a factor, so that the previous reference level settings can be cleared
    robjects.r('r_data$Model <- factor(as.character(r_data$Model))')
    robjects.r('r_data$query_type <- factor(as.character(r_data$query_type))')

    # Then set the reference level required for this function to be contriever and FACTOID
    robjects.r(f'r_data$Model <- relevel(r_data$Model, ref = "{BASE_MODEL}")')
    robjects.r(f'r_data$query_type <- relevel(r_data$query_type, ref = "{BASE_QUERY_TYPE}")')

    # Get the updated r_data from the R environment
    r_data = robjects.globalenv['r_data']


    formula = 'ndcg_10 ~ Model * query_type + (1 | dataset_name/query_id)' # This is a nested structure, which is correct
 

    # Explicitly construct the R formula object
    r_formula = robjects.Formula(formula)
    print(r_formula)
    
    # Use a more robust optimizer setting to help the model converge
    fit = lmerTest.lmer(formula=r_formula, data=r_data, control=robjects.r['ctrl'])
    print("Analysis 1 fit has been fitted successfully")
    
    # Check the convergence status (need to register fit to the R environment first)
    robjects.globalenv['fit'] = fit
    convergence_warnings = robjects.r('isSingular(fit)')
    if convergence_warnings[0]:
        print("⚠️  Warning: Model may be singular (boundary fit)")
    
    # Type III ANOVA test the significance of the fixed effects
    # Original code - here the main effect is "the effect at the reference level"
    # print("\n=== Type III ANOVA for Fixed Effects (at reference level) ===")
    # anova_result = robjects.r('anova(fit, type = 3, ddf = "Satterthwaite")')
    # print(anova_result) 

    # New code - here the true marginal main effect is
    print("\n=== True Marginal Main Effects (averaged across other factor levels) ===")
    joint_tests_result = robjects.r('emmeans::joint_tests(fit)')
    print(joint_tests_result)

    
    
    # Method 1: Calculate the marginal mean of all interaction terms
    print("\n=== Calculate the marginal mean ===")
    specs = robjects.Formula('~ Model * query_type')  # Adjust the actual factors according to your data
    emm = emmeans.emmeans(robjects.globalenv['fit'], specs=specs)
    print(emm)

    # Store emm in the R environment
    # robjects.globalenv['emm'] = emm
 

    # print("\n=== Test the significance between any models (including bm25) ===")
    
    
    # # Execute through R and return the result
    # pairwise = robjects.r('pairs(emm, adjust = "tukey")')
    # print("Original pairs output:")
    # print(pairwise)
    # pairwise_summary = robjects.r('summary(pairs(emm, adjust = "tukey"), infer = c(TRUE, TRUE))')
    
    # print("\n=== Pairwise model contrasts (Tukey-adjusted) with inference statistics ===")
    # print(pairwise_summary)

       # Print the reference level information
    print("\n=== Reference levels ===")
    print(f"Model reference level: {robjects.r('levels(as.factor(r_data$Model))[1]')[0]}")
    print(f"Query type reference level: {robjects.r('levels(as.factor(r_data$query_type))[1]')[0]}")


    print("\n=== Fixed effects ===")

    fixed_effects = robjects.r['coef'](robjects.r['summary'](fit))
    print(fixed_effects)
    print("\n=== Random effects ===")
    random_effects = robjects.r['VarCorr'](fit)
    print(random_effects)

    return fit

def analysis_model_with_task_type_mixedlm(r_data):
    # Analysis 2a:
    # C(Model) * C(task_type) + (1 | dataset_name/query_id)
    print("Analysis 2a:")
    
    # Register r_data to the R global environment, so that R code can access it
    robjects.globalenv['r_data'] = r_data

    # Recreate the factors, ensure that the reference level is cleared
    # First convert to a character type, then convert to a factor, so that the previous reference level settings can be cleared
    robjects.r('r_data$Model <- factor(as.character(r_data$Model))')
    robjects.r('r_data$task_type <- factor(as.character(r_data$task_type))')
    
    # Then set the reference level required for this function to be contriever
    robjects.r(f'r_data$Model <- relevel(r_data$Model, ref = "{BASE_MODEL}")') #contriever
    robjects.r(f'r_data$task_type <- relevel(r_data$task_type, ref = "{BASE_TASK_TYPE}")')

    # Get the updated r_data from the R environment
    r_data = robjects.globalenv['r_data']



    formula = 'ndcg_10 ~ Model * task_type + (1 | dataset_name/query_id)'
 


    r_formula = robjects.Formula(formula)

    # Use a more robust optimizer setting to help the model converge
    fit = lmerTest.lmer(formula=r_formula, data=r_data, control=robjects.r['ctrl'])
    print("Analysis 2a fit has been fitted successfully")
    
    # Check the convergence status (need to register fit to the R environment first)
    robjects.globalenv['fit'] = fit
    convergence_warnings = robjects.r('isSingular(fit)')
    if convergence_warnings[0]:
        print("⚠️  Warning: Model may be singular (boundary fit)")

    # Type III ANOVA test the significance of the fixed effects
    # Original code - here the main effect is "the effect at the reference level"
    # print("\n=== Type III ANOVA for Fixed Effects (at reference level) ===")
    # anova_result = robjects.r('anova(fit, type = 3, ddf = "Satterthwaite")')
    # print(anova_result)

    # New code - here the true marginal main effect is
    print("\n=== True Marginal Main Effects (averaged across other factor levels) ===")
    joint_tests_result = robjects.r('emmeans::joint_tests(fit)')
    print(joint_tests_result)

    # Method 1: Calculate the marginal mean of all interaction terms
    print("\n=== Calculate the marginal mean ===")
    specs = robjects.Formula('~ Model * task_type')  # 根据你的实际因子调整
    emm = emmeans.emmeans(robjects.globalenv['fit'], specs=specs)
    print(emm)

    # Store emm in the R environment
    # robjects.globalenv['emm'] = emm

    # print("\n=== Test the significance between any models (including bm25) ===")

    
    # # Execute through R and return the result
    # pairwise = robjects.r('pairs(emm, adjust = "tukey")')
    # print("Original pairs output:")
    # print(pairwise)
    # pairwise_summary = robjects.r('summary(pairs(emm, adjust = "tukey"), infer = c(TRUE, TRUE))')
    
    # print("\n=== Pairwise model contrasts (Tukey-adjusted) with inference statistics ===")
    # print(pairwise_summary)


    print("\n=== Reference levels ===")
    print(f"Model reference level: {robjects.r('levels(as.factor(r_data$Model))[1]')[0]}")
    print(f"Task type reference level: {robjects.r('levels(as.factor(r_data$task_type))[1]')[0]}")

    print("\n=== Fixed effects ===")
 
    fixed_effects = robjects.r['coef'](robjects.r['summary'](fit))
    print(fixed_effects)
    print("\n=== Random effects ===")
    random_effects = robjects.r['VarCorr'](fit)
    print(random_effects)
    return fit

def analysis_model_with_corpus_type_mixedlm(r_data):
    # Analysis 2b:
    # C(Model) * C(corpus_type) + (1 | dataset_name/query_id)
    print("Analysis 2b:")
    
    # Register r_data to the R global environment, so that R code can access it
    robjects.globalenv['r_data'] = r_data

    # Recreate the factors, ensure that the reference level is cleared
    # First convert to a character type, then convert to a factor, so that the previous reference level settings can be cleared
    robjects.r('r_data$Model <- factor(as.character(r_data$Model))')
    robjects.r('r_data$corpus_type <- factor(as.character(r_data$corpus_type))')
    
    # Then set the reference level required for this function to be contriever
    robjects.r(f'r_data$Model <- relevel(r_data$Model, ref = "{BASE_MODEL}")') #contriever
    robjects.r(f'r_data$corpus_type <- relevel(r_data$corpus_type, ref = "{BASE_CORPUS_TYPE}")')

    # Get the updated r_data from the R environment
    r_data = robjects.globalenv['r_data']

    formula = 'ndcg_10 ~ Model * corpus_type + (1 | dataset_name/query_id)'
 
    r_formula = robjects.Formula(formula)

    # Use a more robust optimizer setting to help the model converge
    fit = lmerTest.lmer(formula=r_formula, data=r_data, control=robjects.r['ctrl'])
    print("Analysis 2b fit has been fitted successfully")
    
    # Check the convergence status (need to register fit to the R environment first)
    robjects.globalenv['fit'] = fit
    convergence_warnings = robjects.r('isSingular(fit)')
    if convergence_warnings[0]:
        print("⚠️  Warning: Model may be singular (boundary fit)")
    # Extract the random effects (random intercept)
    print("\n=== Random Intercepts for dataset_name (Sanity Check) ===")
    random_effects = robjects.r('ranef(fit)$dataset_name')
    print(random_effects)
    # Or more directly view the random intercept of Quora
    print("\n=== Quora random intercept estimate ===")
    robjects.r('''
    re <- ranef(fit)$dataset_name
    if ("quora" %in% rownames(re)) {
        cat("Quora random intercept:", re["quora", "(Intercept)"], "\n")
    } else {
        print("Available dataset names:")
        print(rownames(re))
    }
    ''')

    # Type III ANOVA test the significance of the fixed effects
    # Original code - here the main effect is "the effect at the reference level"
    print("\n=== Type III ANOVA for Fixed Effects (at reference level) ===")
    anova_result = robjects.r('anova(fit, type = 3, ddf = "Satterthwaite")')
    print(anova_result)

    # New code - here the true marginal main effect is
    print("\n=== True Marginal Main Effects (averaged across other factor levels) ===")
    joint_tests_result = robjects.r('emmeans::joint_tests(fit)')
    print(joint_tests_result)

    # Method 1: Calculate the marginal mean of all interaction terms
    print("\n=== Calculate the marginal mean ===")
    specs = robjects.Formula('~ Model * corpus_type')  # Adjust the actual factors according to your data
    emm = emmeans.emmeans(robjects.globalenv['fit'], specs=specs)
    print(emm)

    # Store emm in the R environment
    # robjects.globalenv['emm'] = emm

    # print("\n=== Test the significance between any models (including bm25) ===")
    
    # # Execute through R and return the result
    # pairwise = robjects.r('pairs(emm, adjust = "tukey")')
    # print("Original pairs output:")
    # print(pairwise)
    # pairwise_summary = robjects.r('summary(pairs(emm, adjust = "tukey"), infer = c(TRUE, TRUE))')
    
    # print("\n=== Pairwise model contrasts (Tukey-adjusted) with inference statistics ===")
    # print(pairwise_summary)


    print("\n=== Reference levels ===")
    print(f"Model reference level: {robjects.r('levels(as.factor(r_data$Model))[1]')[0]}")
    print(f"Corpus type reference level: {robjects.r('levels(as.factor(r_data$corpus_type))[1]')[0]}")

    print("\n=== Fixed effects ===")
    fixed_effects = robjects.r['coef'](robjects.r['summary'](fit))
    print(fixed_effects)
    print("\n=== Random effects ===")
    random_effects = robjects.r['VarCorr'](fit)
    print(random_effects)
    
    return fit


def analysis():
    df_long = read_data()
    print("Data Sample:")
    print(df_long.head())
    print(df_long.columns)

    # Use context to convert pandas -> R data.frame
    with conversion.localconverter(robjects.default_converter + pandas2ri.converter):
        r_data = conversion.py2rpy(df_long)
    print("Data has been converted to R data.frame")
    
    # Set the optimizer control parameters in the global environment, for all analysis functions to use
    # Use the bobyqa optimizer, increase the maximum number of iterations, to help the model converge
    robjects.r('ctrl <- lme4::lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 500000))')
    
    # Enable exact degree of freedom calculation (for large sample data)
    # According to the warning information, the number of observations is 326493, set the limit to be greater than or equal to this value
    # Note: This will significantly increase the calculation time and memory usage
    # print("\n⚠️  Enable exact degree of freedom calculation (may significantly increase calculation time)...")
    # robjects.r('emmeans::emm_options(pbkrtest.limit = 350000)')  # Set to slightly greater than the number of observations, leaving some margin
    # robjects.r('emmeans::emm_options(lmerTest.limit = 350000)')  # Set to slightly greater than the number of observations, leaving some margin
    # print("✅ emm_options has been set successfully")
    
    fit = analysis_model_with_query_type_mixedlm(r_data)
    print("Analysis 1   has been finished successfully")
    fit = analysis_model_with_task_type_mixedlm(r_data)
    print("Analysis 2a has been finished successfully")
    
    # ============ Full data analysis ============
    print("\n" + "="*60)
    print("=== Analysis 1: using full data (including Quora) ===")
    print("="*60)
    fit = analysis_model_with_corpus_type_mixedlm(r_data)
    print("Analysis 2b (full data) has been finished successfully")
    
    # ============ Analysis without Quora ============
    print("\n" + "="*60)
    print("=== Analysis 2: remove Quora data (Sanity Check) ===")
    print("="*60)
    
    # Filter out Quora data
    df_long_no_quora = df_long[df_long['dataset_name'] != 'quora'].reset_index(drop=True)
    print(f"Original data rows: {len(df_long)}")
    print(f"Data rows after removing Quora: {len(df_long_no_quora)}")
    print(f"Quora data占比: {(len(df_long) - len(df_long_no_quora)) / len(df_long) * 100:.2f}%")
    
    # Convert to R data.frame
    with conversion.localconverter(robjects.default_converter + pandas2ri.converter):
        r_data_no_quora = conversion.py2rpy(df_long_no_quora)
    
    fit_no_quora = analysis_model_with_corpus_type_mixedlm(r_data_no_quora)
    print("Analysis 2b (remove Quora) has been finished successfully")
    
    print("\n" + "="*60)
    print("=== Comparison completed: please compare the changes in the EMM of Online Community in the two analyses ===")
    print("="*60)

if __name__ == "__main__":
    analysis()
