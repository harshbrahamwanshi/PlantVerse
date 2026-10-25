"""
PlantVerse — Plant Pathology & Disease Recommendations Knowledge Base
====================================================================
Contains verified botanical, pathological, and agronomic guidelines for all 38
classes of the PlantVillage dataset (referenced from the Plant Disease Prediction model).

All recommendations are compiled for informational and educational guidance.
"""

from typing import Any, Dict, List

# Standard 38 PlantVillage Classes mapped to detailed pathological data
DISEASE_DATABASE: Dict[str, Dict[str, Any]] = {
    # ---------------- Apple ----------------
    "Apple Scab": {
        "crop": "Apple",
        "condition": "Apple Scab",
        "health_status": "Diseased",
        "pathogen": "Venturia inaequalis (Fungus)",
        "severity": "Moderate to High",
        "symptoms": "Olive-green to velvety dark brown circular lesions on leaves and fruit. Leaves may twist, pucker, and drop prematurely.",
        "recommendations": [
            "Rake and destroy fallen leaves in autumn to reduce overwintering fungal spores.",
            "Prune tree canopies during dormancy to promote sunlight penetration and rapid leaf drying.",
            "Apply approved organic sulfur or copper-based preventative fungicide at green tip to petal fall stages.",
            "Avoid overhead sprinkler irrigation that keeps leaves wet for prolonged periods."
        ],
        "preventive_measures": [
            "Select scab-resistant apple cultivars (e.g., Liberty, Freedom, Enterprise, Prima).",
            "Maintain wide tree spacing to allow maximum aerodynamic air movement.",
            "Apply a layer of organic mulch around the dripline without touching the trunk."
        ]
    },
    "Apple with Black Rot": {
        "crop": "Apple",
        "condition": "Black Rot",
        "health_status": "Diseased",
        "pathogen": "Botryosphaeria obtusa (Fungus)",
        "severity": "High",
        "symptoms": "Frogeye leaf spots (circular brown spots with purple margins), sunken bark cankers, and firm black mummified rot on fruit.",
        "recommendations": [
            "Prune out dead wood, cankers, and mummified fruit at least 6-8 inches below infected tissue.",
            "Disinfect pruning shears with 70% isopropyl alcohol between every cut.",
            "Remove and burn or deeply bury all infected prunings and fallen debris.",
            "Apply protective fungicide sprays beginning at bud break and continuing through cover sprays."
        ],
        "preventive_measures": [
            "Prevent insect punctures and mechanical wounds that serve as fungal entry sites.",
            "Keep trees vigorous with balanced fertilization, avoiding excess nitrogen.",
            "Maintain diligent orchard sanitation throughout every growing season."
        ]
    },
    "Cedar Apple Rust": {
        "crop": "Apple",
        "condition": "Cedar Apple Rust",
        "health_status": "Diseased",
        "pathogen": "Gymnosporangium juniperi-virginianae (Heteroecious Fungus)",
        "severity": "Moderate",
        "symptoms": "Bright yellow-orange or reddish spots on upper leaf surfaces. Later produces tube-like fungal aecia on leaf undersides.",
        "recommendations": [
            "Inspect nearby Eastern Redcedar or Juniper trees and prune out galls before spring rains.",
            "Remove nearby volunteer juniper seedlings within 1-2 miles if feasible.",
            "Apply targeted protective rust fungicides (e.g., myclobutanil or sulfur) from pink bud through petal fall.",
            "Prune apple canopies to ensure good air circulation."
        ],
        "preventive_measures": [
            "Plant resistant apple varieties (e.g., Redfree, Liberty, William's Pride).",
            "Avoid planting susceptible apples within 500 feet of ornamental juniper species.",
            "Monitor weather forecasts during wet spring periods when spore dissemination peaks."
        ]
    },
    "Healthy Apple": {
        "crop": "Apple",
        "condition": "Healthy Foliage",
        "health_status": "Healthy",
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Vibrant green leaves, uniform margins, well-defined venation, and no visible necrotic lesions or chlorosis.",
        "recommendations": [
            "Continue standard orchard maintenance, weed control, and balanced seasonal irrigation.",
            "Perform regular preventative scouting for early signs of foliar stress or pest vectors.",
            "Apply light organic compost around tree dripline to sustain beneficial mycorrhizal root networks."
        ],
        "preventive_measures": [
            "Maintain winter pruning to sustain open vase canopy architecture.",
            "Ensure soil pH remains between 6.0 and 7.0 for optimal micronutrient uptake."
        ]
    },

    # ---------------- Blueberry ----------------
    "Healthy Blueberry Plant": {
        "crop": "Blueberry",
        "condition": "Healthy Foliage",
        "health_status": "Healthy",
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Glossy ovate foliage, vigorous new shoot flushes, and absence of stem dieback or leaf mottling.",
        "recommendations": [
            "Maintain acidic soil pH (4.5 to 5.2) using elemental sulfur or peat moss.",
            "Ensure consistent shallow root moisture using drip irrigation and pine bark mulch.",
            "Apply ammonium sulfate fertilizer in early spring as recommended by soil tests."
        ],
        "preventive_measures": [
            "Replenish 3-4 inches of acidic organic mulch annually.",
            "Avoid high nitrate fertilizers or unbuffered alkaline municipal tap water."
        ]
    },

    # ---------------- Cherry ----------------
    "Cherry with Powdery Mildew": {
        "crop": "Cherry",
        "condition": "Powdery Mildew",
        "health_status": "Diseased",
        "pathogen": "Podosphaera clandestina (Fungus)",
        "severity": "Moderate",
        "symptoms": "White to grayish powdery fungal patches on young leaves and terminal shoots. Leaves may curl upward, distort, and blister.",
        "recommendations": [
            "Prune severely infected shoot tips and suckers to eliminate primary inoculum.",
            "Apply horticultural oils, neem extract, or potassium bicarbonate upon initial symptom onset.",
            "Improve sunlight penetration by thinning overcrowded interior branches.",
            "Avoid late-season excessive nitrogen fertilization that stimulates tender susceptible growth."
        ],
        "preventive_measures": [
            "Space trees properly to encourage wind movement.",
            "Utilize preventative bio-fungicides (e.g., Bacillus subtilis) in early spring flushes."
        ]
    },
    "Healthy Cherry Plant": {
        "crop": "Cherry",
        "condition": "Healthy Foliage",
        "health_status": "Healthy",
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Clean serrated leaves, bright green luster, sturdy petioles, and robust vegetative shoots.",
        "recommendations": [
            "Continue regular monitoring for leaf curl and cherry fruit fly activity.",
            "Water deeply at the base during fruit set and dry spells.",
            "Apply dormant copper or lime-sulfur wash before winter bud break."
        ],
        "preventive_measures": [
            "Maintain weed-free tree basins to deter rodent and aphid vectors.",
            "Perform annual summer pruning after harvest to maximize interior light."
        ]
    },

    # ---------------- Corn (Maize) ----------------
    "Corn (Maize) with Cercospora and Gray Leaf Spot": {
        "crop": "Corn (Maize)",
        "condition": "Gray Leaf Spot (Cercospora)",
        "health_status": "Diseased",
        "pathogen": "Cercospora zeae-maydis (Fungus)",
        "severity": "High",
        "symptoms": "Rectangular, tan-to-gray lesions running parallel to leaf veins, bounded by vein margins. Severe cases cause blighting.",
        "recommendations": [
            "Rotate crops for at least 1-2 years with non-host species (e.g., soybean, legumes).",
            "Till under or shred infected crop residue where soil conservation practices permit.",
            "Apply strobilurin or triazole fungicides at tassel emergence if weather favors disease.",
            "Ensure optimal plant nutrition, avoiding potassium deficiency."
        ],
        "preventive_measures": [
            "Plant corn hybrids rated high for Gray Leaf Spot resistance.",
            "Maintain moderate plant densities to reduce canopy relative humidity."
        ]
    },
    "Corn (Maize) with Common Rust": {
        "crop": "Corn (Maize)",
        "condition": "Common Rust",
        "health_status": "Diseased",
        "pathogen": "Puccinia sorghi (Fungus)",
        "severity": "Moderate",
        "symptoms": "Golden-brown to cinnamon-brown powdery pustules on both upper and lower leaf surfaces, rupturing epidermis.",
        "recommendations": [
            "Scout lower canopy regularly; apply foliar fungicide if pustules appear before silking on susceptible hybrids.",
            "Monitor air temperature and humidity; rust development slows significantly in hot, dry conditions.",
            "Maintain adequate soil fertility to prevent plant stress."
        ],
        "preventive_measures": [
            "Select corn hybrids possessing genetic resistance (Rp genes).",
            "Plant early in the season to avoid late-season airborne spore migrations."
        ]
    },
    "Corn (Maize) with Northern Leaf Blight": {
        "crop": "Corn (Maize)",
        "condition": "Northern Corn Leaf Blight",
        "health_status": "Diseased",
        "pathogen": "Exserohilum turcicum (Fungus)",
        "severity": "High",
        "symptoms": "Large, elongated cigar-shaped grayish-green to tan lesions (1 to 6 inches long) across leaf blades.",
        "recommendations": [
            "Scout canopy during grain fill; apply targeted systemic fungicide if lesions reach ear leaf.",
            "Practice deep tillage to incorporate crop residue into the soil in conventional fields.",
            "Implement a minimum 2-year crop rotation out of maize."
        ],
        "preventive_measures": [
            "Deploy resistant hybrid cultivars featuring Ht resistance genes.",
            "Manage field drainage to prevent localized water stagnation."
        ]
    },
    "Healthy Corn (Maize) Plant": {
        "crop": "Corn (Maize)",
        "condition": "Healthy Foliage",
        "health_status": "Healthy",
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Broad, arching dark green leaves with clean parallel venation and robust stalk development.",
        "recommendations": [
            "Continue standard sidedress nitrogen application according to growth stage.",
            "Monitor soil moisture levels during critical pollination and grain-filling windows.",
            "Maintain integrated pest scouting for fall armyworm and corn borer."
        ],
        "preventive_measures": [
            "Adhere to balanced crop nutrition and rotation cycles.",
            "Ensure seed treatment protection against soilborne seedling damping-off."
        ]
    },

    # ---------------- Grape ----------------
    "Grape with Black Rot": {
        "crop": "Grape",
        "condition": "Black Rot",
        "health_status": "Diseased",
        "pathogen": "Guignardia bidwellii (Fungus)",
        "severity": "High",
        "symptoms": "Small brown circular leaf spots with dark borders and black pycnidia specks; fruit shrivels into hard black mummies.",
        "recommendations": [
            "Remove and destroy all mummified grape clusters from vines and the vineyard floor during winter.",
            "Prune grapevines to promote open canopy and maximize airflow across leaf surfaces.",
            "Apply preventative fungicides from early shoot growth through 4-5 weeks post-bloom.",
            "Mow vineyard rows regularly to minimize ambient humidity around low-hanging clusters."
        ],
        "preventive_measures": [
            "Plant resistant grape varieties where climate exhibits warm, humid springs.",
            "Train vines on high trellis systems to elevate foliage above ground moisture."
        ]
    },
    "Grape with Esca (Black Measles)": {
        "crop": "Grape",
        "condition": "Esca / Black Measles",
        "health_status": "Diseased",
        "pathogen": "Fomitiporia / Phaeomoniella fungal complex",
        "severity": "Critical",
        "symptoms": "'Tiger-stripe' chlorotic and necrotic patterns between leaf veins; berries develop speckled dark spots and wilt.",
        "recommendations": [
            "Mark symptomatic vines in summer for targeted remedial pruning or trunk renewal.",
            "Delay pruning until late winter when wound susceptibility is lowest.",
            "Treat large pruning cuts immediately with wound sealants or Trichoderma biocontrols.",
            "Remove severely debilitated vines to protect neighboring plants."
        ],
        "preventive_measures": [
            "Sanitize pruning blades frequently when moving between vineyard blocks.",
            "Avoid making large flush cuts on cordons that create deep wood entry conduits."
        ]
    },
    "Grape with Isariopsis Leaf Spot": {
        "crop": "Grape",
        "condition": "Isariopsis Leaf Blight",
        "health_status": "Diseased",
        "pathogen": "Pseudocercospora cladosporioides (Fungus)",
        "severity": "Moderate",
        "symptoms": "Irregular dark brown necrotic lesions with faint reddish borders on mature foliage; premature defoliation.",
        "recommendations": [
            "Collect and destroy fallen leaf debris post-harvest.",
            "Prune lower canopy leaves (shoot thinning and leaf pulling) around cluster zones.",
            "Apply copper-based or dithiocarbamate fungicide sprays if disease pressure is elevated.",
            "Ensure proper vine trellising to prevent canopy crowding."
        ],
        "preventive_measures": [
            "Implement balanced potassium and phosphorus fertilization.",
            "Avoid sprinkler irrigation hitting vine canopies."
        ]
    },
    "Healthy Grape Plant": {
        "crop": "Grape",
        "condition": "Healthy Foliage",
        "health_status": "Healthy",
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Expansive palmate green leaves, intact leaf margins, vigorous tendril growth, and clean berry clusters.",
        "recommendations": [
            "Maintain disciplined canopy shoot positioning and fruit zone leaf pulling.",
            "Monitor soil water tension using tensiometers for deficit irrigation strategies.",
            "Maintain protective sulfur dusting during mildew-prone intervals."
        ],
        "preventive_measures": [
            "Annual dormant pruning to balance crop load with vegetative capacity.",
            "Scout for grape leafhopper and berry moth vectors regularly."
        ]
    },

    # ---------------- Orange / Citrus ----------------
    "Orange with Citrus Greening": {
        "crop": "Orange / Citrus",
        "condition": "Citrus Greening (Huanglongbing)",
        "health_status": "Diseased",
        "pathogen": "Candidatus Liberibacter asiaticus (Bacterium vectored by Asian Citrus Psyllid)",
        "severity": "Critical",
        "symptoms": "Asymmetrical blotchy leaf mottle (crossing leaf veins), yellow shoots, lopsided bitter fruit with green bottoms.",
        "recommendations": [
            "Report suspected sightings to local agricultural extension or quarantine authorities.",
            "Control Asian citrus psyllid insect vectors using integrated pest management or horticultural oils.",
            "Supply supplemental micronutrient foliar sprays (zinc, manganese, iron) to support tree vigor.",
            "Remove and destroy severely affected non-productive trees to reduce bacterial reservoirs."
        ],
        "preventive_measures": [
            "Plant only certified disease-free nursery stock from screenhouse facilities.",
            "Maintain strict regional quarantine protocols against unverified citrus budwood."
        ]
    },

    # ---------------- Peach ----------------
    "Peach with Bacterial Spot": {
        "crop": "Peach",
        "condition": "Bacterial Spot",
        "health_status": "Diseased",
        "pathogen": "Xanthomonas campestris pv. pruni (Bacterium)",
        "severity": "High",
        "symptoms": "Water-soaked polygonal leaf spots turning dark purple or brown, dropping out to leave a 'shot-hole' appearance; fruit pitting.",
        "recommendations": [
            "Apply dormant copper sprays in late autumn and late winter before bud swell.",
            "Prune during dry weather to prevent bacterial spread across wet cuts.",
            "Avoid high nitrogen fertilization which produces excessively succulent vulnerable tissue.",
            "Apply protective oxytetracycline or low-rate copper during the growing season where registered."
        ],
        "preventive_measures": [
            "Choose resistant peach cultivars (e.g., Redhaven, Candor, Biscoe).",
            "Plant windbreaks around sandy sites to reduce windblown sand abrasions that introduce bacteria."
        ]
    },
    "Healthy Peach Plant": {
        "crop": "Peach",
        "condition": "Healthy Foliage",
        "health_status": "Healthy",
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Long, lanceolate deep green leaves with glossy surfaces, healthy red-tinted new wood, and sturdy branch growth.",
        "recommendations": [
            "Maintain open center (vase) tree architecture through annual winter pruning.",
            "Ensure regular irrigation during fruit swell while avoiding crown saturation.",
            "Apply dormant preventative spray for peach leaf curl before bud swell."
        ],
        "preventive_measures": [
            "Thin fruit clusters early to avoid branch breakage and tree stress.",
            "Paint lower trunks with white latex paint to prevent winter sunscald."
        ]
    },

    # ---------------- Bell Pepper ----------------
    "Bell Pepper with Bacterial Spot": {
        "crop": "Bell Pepper",
        "condition": "Bacterial Spot",
        "health_status": "Diseased",
        "pathogen": "Xanthomonas campestris pv. vesicatoria (Bacterium)",
        "severity": "High",
        "symptoms": "Small, water-soaked brown spots on leaves, becoming necrotic and scab-like. Severe infections trigger leaf yellowing and dropping.",
        "recommendations": [
            "Avoid handling plants or cultivating fields when foliage is wet from dew or rain.",
            "Apply fixed copper mixed with mancozeb to slow bacterial spread in early stages.",
            "Remove and destroy heavily diseased individual plants.",
            "Switch immediately to drip irrigation to keep pepper leaves dry."
        ],
        "preventive_measures": [
            "Use certified pathogen-free hot water-treated seed or transplants.",
            "Implement a minimum 2-3 year crop rotation without solanaceous crops (tomato, potato, eggplant).",
            "Select pepper cultivars possessing bacterial spot resistance genes."
        ]
    },
    "Healthy Bell Pepper Plant": {
        "crop": "Bell Pepper",
        "condition": "Healthy Foliage",
        "health_status": "Healthy",
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Deep green glossy leaves with smooth margins, sturdy branching, and vigorous white blossoms.",
        "recommendations": [
            "Maintain steady soil moisture to prevent blossom end rot (calcium deficiency).",
            "Apply balanced fertilizer rich in phosphorus and potassium at first flowering.",
            "Provide light support stakes to keep heavy pepper fruit off the soil surface."
        ],
        "preventive_measures": [
            "Apply straw or plastic mulch to suppress weeds and stabilize soil temperature.",
            "Scout for aphid colonies and pepper weevil vectors periodically."
        ]
    },

    # ---------------- Potato ----------------
    "Potato with Early Blight": {
        "crop": "Potato",
        "condition": "Early Blight",
        "health_status": "Diseased",
        "pathogen": "Alternaria solani (Fungus)",
        "severity": "Moderate to High",
        "symptoms": "Dark brown to black concentric target-like rings on older leaves, surrounded by yellow chlorotic margins.",
        "recommendations": [
            "Prune or remove severely infected lower foliage to reduce soil-splashed spores.",
            "Apply preventive biofungicides (Bacillus) or protective copper sprays every 7-10 days in humid conditions.",
            "Maintain adequate nitrogen and potassium fertility; stressed or senescing plants are much more susceptible.",
            "Avoid overhead irrigation; water early in the morning so leaves dry rapidly."
        ],
        "preventive_measures": [
            "Practice 3-year crop rotation avoiding other nightshade family members.",
            "Hill potatoes thoroughly to prevent spores from washing into soil and infecting tubers.",
            "Allow tubers to fully mature and cure skin before post-harvest storage."
        ]
    },
    "Potato with Late Blight": {
        "crop": "Potato",
        "condition": "Late Blight",
        "health_status": "Diseased",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "severity": "Critical",
        "symptoms": "Large, irregular water-soaked dark brown or black lesions with pale green borders; white fuzzy growth under leaves in humid air.",
        "recommendations": [
            "Immediately destroy or bury infected foliage; late blight can spread across miles via airborne sporangia.",
            "Apply specialized oomycete fungicides (e.g., mandipropamid or copper) at the earliest warning in the region.",
            "Kill vines 2-3 weeks prior to harvest to prevent tuber contact with live spores.",
            "Do not harvest when soil is wet to prevent tuber rot in storage."
        ],
        "preventive_measures": [
            "Plant only certified disease-free seed tubers.",
            "Eliminate all cull piles and volunteer potatoes in the vicinity before planting.",
            "Choose resistant varieties (e.g., Kennebec, Defender, Elba)."
        ]
    },
    "Healthy Potato Plant": {
        "crop": "Potato",
        "condition": "Healthy Foliage",
        "health_status": "Healthy",
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Lush, bushy compound leaves with rich green coloration, erect stems, and vigorous flower clusters.",
        "recommendations": [
            "Keep potato tubers deeply hilled to shield them from sunlight (greening) and potato tuber moth.",
            "Provide 1-2 inches of water per week during tuber bulking stage.",
            "Inspect underside of leaves weekly for Colorado potato beetle egg clusters."
        ],
        "preventive_measures": [
            "Rotate crops annually with brassicas or grasses.",
            "Use certified disease-free seed potatoes."
        ]
    },

    # ---------------- Raspberry ----------------
    "Healthy Raspberry Plant": {
        "crop": "Raspberry",
        "condition": "Healthy Foliage",
        "health_status": "Healthy",
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Pinnately compound serrate leaves, bright green appearance, sturdy upright primocanes, and healthy bramble growth.",
        "recommendations": [
            "Prune spent floricanes to the ground immediately after summer harvest.",
            "Maintain trellis wires to keep canes upright and well aerated.",
            "Apply woodchip mulch and ensure moist, well-draining soil."
        ],
        "preventive_measures": [
            "Avoid planting raspberries in soil where tomatoes, peppers, or potatoes grew recently (Verticillium risk).",
            "Maintain 3-4 feet row spacing for air movement."
        ]
    },

    # ---------------- Soybean ----------------
    "Healthy Soybean Plant": {
        "crop": "Soybean",
        "condition": "Healthy Foliage",
        "health_status": "Healthy",
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Trifoliate leaves with vibrant green color, strong trifoliate symmetry, upright canopy, and active root nodulation.",
        "recommendations": [
            "Maintain balanced phosphorus and potassium fertility according to soil test recommendations.",
            "Scout fields periodically for bean leaf beetle and soybean aphid populations.",
            "Ensure effective weed canopy closure through proper row spacing."
        ],
        "preventive_measures": [
            "Inoculate seed with Bradyrhizobium japonicum before planting.",
            "Practice annual crop rotation with corn or small grains."
        ]
    },

    # ---------------- Squash ----------------
    "Squash with Powdery Mildew": {
        "crop": "Squash",
        "condition": "Powdery Mildew",
        "health_status": "Diseased",
        "pathogen": "Podosphaera xanthii / Erysiphe cichoracearum (Fungi)",
        "severity": "Moderate to High",
        "symptoms": "Distinct white talcum-like powdery spots spreading over upper and lower leaf surfaces, leading to yellowing and crisping.",
        "recommendations": [
            "Apply organic horticultural oils, potassium bicarbonate, or neem oil at first sign of powdery patches.",
            "Prune off heavily covered older crown leaves to open up the canopy center.",
            "Water exclusively at soil level using soaker hoses or drip lines.",
            "Ensure full sun exposure (minimum 6-8 hours daily)."
        ],
        "preventive_measures": [
            "Select powdery mildew-resistant squash varieties (e.g., Success PM, Dunja).",
            "Provide ample vine spacing (at least 4-6 feet between rows).",
            "Apply preventative sulfur or bio-fungicide sprays before canopy closure."
        ]
    },

    # ---------------- Strawberry ----------------
    "Strawberry with Leaf Scorch": {
        "crop": "Strawberry",
        "condition": "Leaf Scorch",
        "health_status": "Diseased",
        "pathogen": "Diplocarpon earlianum (Fungus)",
        "severity": "Moderate",
        "symptoms": "Small, irregular dark purple to brownish spots across leaves; spots coalesce causing whole leaf to turn red/brown and scorch.",
        "recommendations": [
            "Mow or clip back diseased strawberry foliage after harvest during renovation.",
            "Remove and destroy heavily scorched leaves from the patch.",
            "Improve row aeration by thinning overcrowded runner plants.",
            "Apply protective fungicide (copper or captan) early in spring if historical pressure is high."
        ],
        "preventive_measures": [
            "Plant resistant strawberry cultivars (e.g., Honeoye, Allstar).",
            "Establish raised beds with clean straw mulch to keep leaves off bare soil."
        ]
    },
    "Healthy Strawberry Plant": {
        "crop": "Strawberry",
        "condition": "Healthy Foliage",
        "health_status": "Healthy",
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Deep green trifoliate leaves, robust crowns, clean petioles, and healthy white blossoms.",
        "recommendations": [
            "Tuck clean straw mulch under developing fruit to prevent soil contact and gray mold.",
            "Irrigate regularly at root level, providing 1 inch of water weekly.",
            "Remove excessive runners to channel energy into crown and fruit development."
        ],
        "preventive_measures": [
            "Renovate June-bearing strawberry beds annually post-harvest.",
            "Rotate strawberry plantings every 3-4 years to fresh garden beds."
        ]
    },

    # ---------------- Tomato ----------------
    "Tomato with Bacterial Spot": {
        "crop": "Tomato",
        "condition": "Bacterial Spot",
        "health_status": "Diseased",
        "pathogen": "Xanthomonas campestris pv. vesicatoria (Bacterium)",
        "severity": "High",
        "symptoms": "Small dark brown to black water-soaked leaf spots with greasy appearance; centers may dry and tear; fruit shows rough scab spots.",
        "recommendations": [
            "Do not work in tomato patch when foliage is wet.",
            "Apply fixed copper fungicide combined with mancozeb to protect uninfected leaves.",
            "Remove and destroy severely blighted plants to curb bacterial spread.",
            "Water with drip irrigation only at the plant base."
        ],
        "preventive_measures": [
            "Purchase certified disease-free seeds or hot-water treated seeds.",
            "Rotate beds away from nightshades for a minimum of 2-3 years.",
            "Disinfect stakes, trellises, and plant cages with 10% bleach solution."
        ]
    },
    "Tomato with Early Blight": {
        "crop": "Tomato",
        "condition": "Early Blight",
        "health_status": "Diseased",
        "pathogen": "Alternaria solani (Fungus)",
        "severity": "High",
        "symptoms": "Concentric dark brown 'target-like' rings on older lower leaves, surrounded by yellow chlorotic halos; progressive upward defoliation.",
        "recommendations": [
            "Prune off all infected lower leaves touching or close to the ground using sanitized shears.",
            "Apply organic copper fungicide, chlorothalonil, or biofungicide (Bacillus subtilis) immediately.",
            "Mulch soil heavily (2-3 inches of straw or wood shavings) to prevent rain-splash of soilborne spores.",
            "Avoid overhead watering; irrigate strictly at the base in early morning."
        ],
        "preventive_measures": [
            "Stake or cage tomato plants to maintain upright, airy growth.",
            "Rotate crops annually with non-solanaceous species.",
            "Select early blight-tolerant varieties (e.g., Mountain Magic, Defiant, Jasper)."
        ]
    },
    "Tomato with Late Blight": {
        "crop": "Tomato",
        "condition": "Late Blight",
        "health_status": "Diseased",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "severity": "Critical",
        "symptoms": "Large, irregularly shaped greasy dark brown to purplish lesions on leaves and stems; white fungal fuzz on leaf undersides in humid air.",
        "recommendations": [
            "Immediate intervention required: pull and bag infected plants to prevent airborne spore epidemics.",
            "Do not compost infected plants; solarize or burn where permitted.",
            "Apply preventative targeted late blight fungicides (copper or chlorothalonil) on remaining plants.",
            "Monitor regional late blight tracking maps and agricultural alerts."
        ],
        "preventive_measures": [
            "Plant late blight-resistant cultivars (e.g., Legend, Defiant Ph-R, Mountain Merit).",
            "Ensure full sun and wide plant spacing (at least 3 feet between plants).",
            "Never compost plant waste suspected of harboring Phytophthora."
        ]
    },
    "Tomato with Leaf Mold": {
        "crop": "Tomato",
        "condition": "Leaf Mold",
        "health_status": "Diseased",
        "pathogen": "Passalora fulva (Fungus)",
        "severity": "Moderate to High",
        "symptoms": "Pale green or yellow spots on upper leaf surfaces; olive-green to grayish velvety mold on the corresponding leaf undersides.",
        "recommendations": [
            "Significantly reduce humidity and increase airflow, especially in greenhouse or high-tunnel setups.",
            "Prune lower suckers and dense center foliage to facilitate cross-ventilation.",
            "Apply preventative copper or sulfur fungicides to protect upper leaves.",
            "Avoid wetting foliage during late afternoon or evening waterings."
        ],
        "preventive_measures": [
            "Maintain greenhouse relative humidity below 85% using exhaust fans.",
            "Plant resistant tomato hybrid cultivars (look for Cf gene resistance)."
        ]
    },
    "Tomato with Septoria Leaf Spot": {
        "crop": "Tomato",
        "condition": "Septoria Leaf Spot",
        "health_status": "Diseased",
        "pathogen": "Septoria lycopersici (Fungus)",
        "severity": "High",
        "symptoms": "Numerous small circular spots (1/16 to 1/8 inch) with dark brown borders and pale tan/gray centers containing tiny black pycnidia specks.",
        "recommendations": [
            "Strip off and discard lower diseased leaves at first detection.",
            "Apply thick clean mulch over soil to block fungal spores residing in garden dirt.",
            "Spray with copper-based or biofungicide every 7-10 days throughout wet periods.",
            "Disinfect cages and stakes at the conclusion of the growing season."
        ],
        "preventive_measures": [
            "Stake plants upright to separate foliage from garden soil.",
            "Practice minimum 2-year rotation out of solanaceous vegetables."
        ]
    },
    "Tomato with Spider Mites or Two-spotted Spider Mite": {
        "crop": "Tomato",
        "condition": "Two-Spotted Spider Mites",
        "health_status": "Diseased",
        "pathogen": "Tetranychus urticae (Acari / Pest infestation)",
        "severity": "High",
        "symptoms": "Fine yellow-white stippling on leaf surfaces; leaves turn bronze and dry up; fine silky webbing visible on leaf undersides and branch crotches.",
        "recommendations": [
            "Spray leaf undersides thoroughly with strong blasts of water to dislodge mite colonies.",
            "Apply insecticidal soap, neem oil, or horticultural oil covering all leaf surfaces.",
            "Introduce beneficial predatory mites (e.g., Phytoseiulus persimilis) in enclosed environments.",
            "Keep soil adequately moist; drought-stressed plants are prime targets for rapid mite reproduction."
        ],
        "preventive_measures": [
            "Avoid broad-spectrum synthetic pesticides that destroy natural ladybug and mite predators.",
            "Maintain adequate garden humidity during hot, dry summer spells."
        ]
    },
    "Tomato with Target Spot": {
        "crop": "Tomato",
        "condition": "Target Spot",
        "health_status": "Diseased",
        "pathogen": "Corynespora cassiicola (Fungus)",
        "severity": "Moderate to High",
        "symptoms": "Small brown pinpoint spots enlarging to circular lesions with light brown centers and dark concentric rings; fruit develops sunken lesions.",
        "recommendations": [
            "Prune dense inner foliage to enhance air movement through the tomato canopy.",
            "Apply protective fungicides (e.g., chlorothalonil, copper, or azoxystrobin) according to label instructions.",
            "Avoid overhead irrigation; water at base with soaker hoses.",
            "Remove all plant debris promptly at season end."
        ],
        "preventive_measures": [
            "Implement a 3-year crop rotation.",
            "Avoid excessive nitrogen fertilization that creates dense humid foliage."
        ]
    },
    "Tomato Yellow Leaf Curl Virus": {
        "crop": "Tomato",
        "condition": "Yellow Leaf Curl Virus (TYLCV)",
        "health_status": "Diseased",
        "pathogen": "Begomovirus (Vectored by Sweetpotato Whitefly, Bemisia tabaci)",
        "severity": "Critical",
        "symptoms": "Severe plant stunting, erect cupped leaves with prominent yellow chlorotic margins, reduced leaf size, and flower drop without fruit set.",
        "recommendations": [
            "Immediately remove and bag infected plants; viral infections cannot be cured with chemical sprays.",
            "Control whitefly insect vectors using yellow sticky cards, insecticidal soap, or reflective silver mulch.",
            "Cover young seedlings with fine insect netting (row covers) until flowering.",
            "Control broadleaf weeds around the garden perimeter that act as alternate whitefly hosts."
        ],
        "preventive_measures": [
            "Plant TYLCV-resistant tomato cultivars (e.g., Tycoon, Red Deuce, Charger).",
            "Purchase certified clean transplants grown in insect-screened nursery houses."
        ]
    },
    "Tomato Mosaic Virus": {
        "crop": "Tomato",
        "condition": "Tomato Mosaic Virus (ToMV)",
        "health_status": "Diseased",
        "pathogen": "Tobamovirus (Mechanically transmitted virus)",
        "severity": "High",
        "symptoms": "Mottled light and dark green mosaic patterns on leaves, distortion or 'fern-like' leaves, and uneven ripening of fruit.",
        "recommendations": [
            "Remove and destroy infected plants immediately to prevent sap transmission to adjacent vines.",
            "Wash hands thoroughly with soap and water after handling tobacco products before touching tomato plants.",
            "Disinfect all tools in a 20% nonfat dry milk solution or 10% bleach.",
            "Do not save seeds from infected plants."
        ],
        "preventive_measures": [
            "Plant certified virus-free seeds and ToMV-resistant cultivars (designated with 'T' or 'ToMV').",
            "Avoid smoking or using tobacco products in or near the tomato patch."
        ]
    },
    "Healthy Tomato Plant": {
        "crop": "Tomato",
        "condition": "Healthy Foliage",
        "health_status": "Healthy",
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Deep green compound leaves, vigorous terminal growth, robust stems, and healthy yellow blossoms without mottling or spots.",
        "recommendations": [
            "Continue regular indeterminate tomato suckering and staking.",
            "Maintain deep, uniform watering (1 to 2 inches weekly) to prevent blossom end rot and fruit splitting.",
            "Replenish straw mulch around the root zone to stabilize soil moisture and temperature."
        ],
        "preventive_measures": [
            "Prune lowest 12 inches of leaves once plants reach 3 feet to prevent soil splash.",
            "Inspect weekly for hornworms and aphids under leaves."
        ]
    }
}


# Mapping for raw class labels (from the reference GitHub repository) to standard keys
RAW_LABEL_MAP: Dict[str, str] = {
    "Apple___Apple_scab": "Apple Scab",
    "Apple___Black_rot": "Apple with Black Rot",
    "Apple___Cedar_apple_rust": "Cedar Apple Rust",
    "Apple___healthy": "Healthy Apple",
    "Blueberry___healthy": "Healthy Blueberry Plant",
    "Cherry_(including_sour)___Powdery_mildew": "Cherry with Powdery Mildew",
    "Cherry_(including_sour)___healthy": "Healthy Cherry Plant",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn (Maize) with Cercospora and Gray Leaf Spot",
    "Corn_(maize)___Common_rust_": "Corn (Maize) with Common Rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Corn (Maize) with Northern Leaf Blight",
    "Corn_(maize)___healthy": "Healthy Corn (Maize) Plant",
    "Grape___Black_rot": "Grape with Black Rot",
    "Grape___Esca_(Black_Measles)": "Grape with Esca (Black Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape with Isariopsis Leaf Spot",
    "Grape___healthy": "Healthy Grape Plant",
    "Orange___Haunglongbing_(Citrus_greening)": "Orange with Citrus Greening",
    "Peach___Bacterial_spot": "Peach with Bacterial Spot",
    "Peach___healthy": "Healthy Peach Plant",
    "Pepper,_bell___Bacterial_spot": "Bell Pepper with Bacterial Spot",
    "Pepper,_bell___healthy": "Healthy Bell Pepper Plant",
    "Potato___Early_blight": "Potato with Early Blight",
    "Potato___Late_blight": "Potato with Late Blight",
    "Potato___healthy": "Healthy Potato Plant",
    "Raspberry___healthy": "Healthy Raspberry Plant",
    "Soybean___healthy": "Healthy Soybean Plant",
    "Squash___Powdery_mildew": "Squash with Powdery Mildew",
    "Strawberry___Leaf_scorch": "Strawberry with Leaf Scorch",
    "Strawberry___healthy": "Healthy Strawberry Plant",
    "Tomato___Bacterial_spot": "Tomato with Bacterial Spot",
    "Tomato___Early_blight": "Tomato with Early Blight",
    "Tomato___Late_blight": "Tomato with Late Blight",
    "Tomato___Leaf_Mold": "Tomato with Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Tomato with Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato with Spider Mites or Two-spotted Spider Mite",
    "Tomato___Target_Spot": "Tomato with Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato___healthy": "Healthy Tomato Plant",
}


import json
from pathlib import Path

NO_RECOMMENDATION_MESSAGE = "No recommendation available for this disease."
EXPERT_REVIEW_NOTICE = (
    "Requires expert agronomic review: Guidance is compiled for educational reference. "
    "Consult a certified agronomist or local Krishi Vigyan Kendra (KVK) extension officer before applying treatments."
)

_DATA_DIR = Path(__file__).resolve().parent / "data"
_EXTRA_KB: Dict[str, Dict[str, Any]] = {}
_CROP_CARE_MAP: Dict[str, Dict[str, str]] = {}

try:
    _kb_path = _DATA_DIR / "crop_diseases_kb.json"
    if _kb_path.exists():
        _kb_raw = json.loads(_kb_path.read_text(encoding="utf-8"))
        for _cls in _kb_raw.get("classes", []):
            _EXTRA_KB[_cls["id"]] = _cls
    _pl_path = _DATA_DIR / "plants_metadata.json"
    if _pl_path.exists():
        _pl_raw = json.loads(_pl_path.read_text(encoding="utf-8"))
        for _p in _pl_raw.get("plants", []):
            if _p.get("care"):
                _CROP_CARE_MAP[_p["name"].lower()] = _p["care"]
                if _p.get("pv_crop"):
                    _CROP_CARE_MAP[_p["pv_crop"].lower()] = _p["care"]
except Exception:
    pass

_KEY_TO_RAW_ID: Dict[str, str] = {v: k for k, v in RAW_LABEL_MAP.items()}


def normalize_disease_label(label: str) -> str:
    """Normalize raw model label strings into standard database keys."""
    if not label:
        return "Unknown"
    clean = label.strip()
    if clean in DISEASE_DATABASE:
        return clean
    if clean in RAW_LABEL_MAP:
        return RAW_LABEL_MAP[clean]

    lower_clean = clean.lower().replace("_", " ").replace("-", " ")
    for key in DISEASE_DATABASE:
        if key.lower() == lower_clean:
            return key

    for raw, key in RAW_LABEL_MAP.items():
        if raw.lower().replace("_", " ") == lower_clean:
            return key

    return clean


def get_disease_info(label: str) -> Dict[str, Any]:
    """
    Retrieve verified pathology, health status, treatment, prevention, and care
    for a known plant disease class.
    If the disease is not in the verified knowledge base, returns empty recommendations
    and 'No recommendation available for this disease.' (never invents advice).
    """
    norm_key = normalize_disease_label(label)
    if norm_key in DISEASE_DATABASE:
        data = dict(DISEASE_DATABASE[norm_key])
        raw_id = _KEY_TO_RAW_ID.get(norm_key, label)
        kb_entry = _EXTRA_KB.get(raw_id, {})
        crop_name = data.get("crop", "Plant")
        care_key = crop_name.lower().split("/")[0].split("(")[0].strip()
        care_info = (
            _CROP_CARE_MAP.get(crop_name.lower())
            or _CROP_CARE_MAP.get(care_key)
            or {
                "sun": "Full sun (6–8 hours daily)",
                "water": "Moderate base irrigation; avoid wetting foliage",
                "soil": "Well-drained, fertile soil with organic matter",
            }
        )
        data["key"] = norm_key
        data["raw_id"] = raw_id
        data["known_disease"] = True
        data["chemical_treatments"] = kb_entry.get("chemical", [])
        data["organic_treatments"] = kb_entry.get(
            "organic",
            ["No chemical/organic intervention needed"] if data.get("health_status") == "Healthy" else [],
        )
        data["care"] = care_info
        data["needs_expert_review"] = True
        data["expert_review_notice"] = EXPERT_REVIEW_NOTICE
        data["no_recommendation_message"] = ""
        return data

    # Unknown or unsupported disease class: never fabricate recommendations
    return {
        "crop": "Unknown",
        "condition": label or "Unknown",
        "health_status": "Unknown",
        "pathogen": "N/A",
        "severity": "N/A",
        "symptoms": NO_RECOMMENDATION_MESSAGE,
        "recommendations": [],
        "preventive_measures": [],
        "chemical_treatments": [],
        "organic_treatments": [],
        "care": {},
        "known_disease": False,
        "needs_expert_review": True,
        "expert_review_notice": EXPERT_REVIEW_NOTICE,
        "no_recommendation_message": NO_RECOMMENDATION_MESSAGE,
        "key": label or "Unknown",
    }

