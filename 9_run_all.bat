py 1_extract_treatment_and_control_countries.py
echo "Finished Script 1"
py 2_combine_trade_data.py
echo "Finished Script 2"
py 3_add_dem_rating_for_trade_partners.py
echo "Finished Script 3"
py 4_scm_add_weigths.py
echo "Finished Script 4"