# AlightRAG: Associations Light Your Path to the Answer
## 1. Code
### 1.1. https://github.com/0x0addc001/AlightRAG
## 2. Usage
### 2.1. modify .env: use your api key for llm and embedding
### 2.2. modify your book.txt
### 2.3. modify alightrag_test.py: use the right alightrag.llm package which is applicable for your llm and embedding, modify your query
### 2.4. run python alightrag_test.py
   - hybrid -> lightrag
   - alightrag -> alightrag
   - reasoning -> lightrag+reasoning
   - reflection -> lightrag+reflection
## 3. Demo
### 3.1. Context
```markdown
Passage: The Tangled Web of Centennial City
In the heart of the Arid Basin, Centennial City emerged not as a traditional metropolis but as a series of interconnected, climate-controlled biodomes. Its founding in 2075 was spearheaded by the Kaito Foundation, a consortium of Japanese and Norwegian engineering firms specializing in glacial geo-engineering. The Foundation's first CEO, Elara Vance, famously opposed the city's initial energy design, advocating for a solar-tidal hybrid system. However, the city council, influenced by the powerful mining guild "The Veridian Pact," opted for a volatile but powerful Thorium-Fission Reactor, nicknamed "Prometheus."
The reactor's core containment vessel was constructed using a patented, translucent alloy known as "Ceruleum," synthesized primarily from rare-earth elements mined in the contested Mesabi Trench, located under the former Great Lakes. The sole licensing rights for Ceruleum were held by a reclusive materials scientist, Dr. Aris Thorne, who operated from a research station on Callisto, Jupiter's moon. Thorne's licensing agreement had a unique clause: a 2% royalty on all energy produced using Ceruleum was to be paid directly to the "Ocean Reclamation Trust" (ORT), a non-profit dedicated to rebuilding coral reefs.
The Veridian Pact, seeking to cut costs, secretly subcontracted the mineral extraction for the Ceruleum to an automated drone fleet operated by "Delta-7 Mining," a subsidiary of the larger Kaito Foundation. This created a severe conflict of interest, unbeknownst to the city council. Delta-7's drones used a highly efficient but ecologically damaging sonar-pulse technique to locate deposits, a method explicitly banned under the "Arid Basin Conservation Act of 2060."
In 2088, a minor fault in the Prometheus reactor's secondary cooling system, traced to a brittle junction made of sub-standard Ceruleum, caused a city-wide "Brownout." The subsequent investigation, led by Chief Inspector Maya Petrova, uncovered the illicit mining operation. The scandal, dubbed "The Ceruleum Affair," led to the dissolution of The Veridian Pact and a complete overhaul of the city's energy grid to the solar-tidal system Elara Vance had originally proposed. The settlement fines were used to fund the ORT's largest project: the "Neo-Florida Keys Reef."
```
### 3.2. Multi-Hop Question
```markdown
Multi-Hop Question:
What specific ecological project was ultimately funded as a direct, albeit indirect, consequence of the Kaito Foundation's subsidiary violating a 2060 conservation act during the procurement of materials for Centennial City's primary power source?
```
### 3.3. Ground Truth Answer 
```markdown
Multi-Hop Question:
What specific ecological project was ultimately funded as a direct, albeit indirect, consequence of the Kaito Foundation's subsidiary violating a 2060 conservation act during the procurement of materials for Centennial City's primary power source?
Reasoning Chain Required (This is what the RAG system must deduce):
Hop 1: Identify the city's primary power source (the Thorium-Fission Reactor "Prometheus").
Hop 2: Identify the key material used in that reactor's construction (Ceruleum alloy).
Hop 3: Identify who illegally mined the materials for that alloy and the violation (Delta-7 Mining, a Kaito Foundation subsidiary, used a sonar-pulse technique banned under the Arid Basin Conservation Act of 2060).
Hop 4: Identify the major scandal that resulted from this discovery (The Ceruleum Affair) and its consequences (fines/settlement).
Hop 5: Trace where the settlement money was directed (to the Ocean Reclamation Trust - ORT) and identify the specific project it funded (the "Neo-Florida Keys Reef").
Final Answer: The Neo-Florida Keys Reef project.
This question tests the system's ability to connect entities across the text (Kaito Foundation -> Delta-7 -> Banned Act -> Scandal -> Fines -> ORT -> Specific Project) without any single sentence containing the direct link. It requires synthesis of information from the beginning, middle, and end of the passage.
```
### 3.4. LightRAG Answer
```markdown
The specific ecological project that was ultimately funded as a consequence of the Kaito Foundation's subsidiary, Delta-7 Mining, violating the Arid Basin Conservation Act of 2060 is the "Neo-Florida Keys Reef." This project became the largest initiative supported by the Ocean Reclamation Trust (ORT). The project was funded by settlement fines derived from the scandal known as "The Ceruleum Affair," which involved Delta-7 Mining's illicit mining operations using ecologically damaging sonar-pulse techniques that were banned under the conservation act.
```
### 3.5. AlightRAG Answer 
```markdown
# Answer
The specific ecological project funded as a consequence was the "Neo-Florida Keys Reef."

# Detailed Reasoning

The question asks which ecological project was funded as a result, directly or indirectly, by the violation of a conservation act i
nvolving material procurement for Centennial City's power source. The entities and relationships provided suggest that "Delta-7 Min
ing," a subsidiary of the Kaito Foundation, violated the "Arid Basin Conservation Act of 2060." The illicit operations of Delta-7 Mining were associated with the procurement of "Ceruleum," a crucial material for the "Prometheus" reactor in Centennial City.      

Separate information from document chunks details that the settlement fines from "The Ceruleum Affair" were allocated to fund the "
Neo-Florida Keys Reef," a project managed by the "Ocean Reclamation Trust." Thus, this series of events led to the financing of the Neo-Florida Keys Reef project.

# Reasoning Path Analysis

## Path 1: Delta-7 Mining -> Delta-7 Mining's mining practices violated the Ari... -> Arid Basin Conservation Act of 2060
**Step-by-Step Explanation:**
1. **Starting Point:** Delta-7 Mining is the actor involved in material procurement.
2. **Violation:** Their mining operations violated the Arid Basin Conservation Act of 2060 by employing damaging sonar-pulse techniques.
3. **Regulatory Relation:** The breach of this act directly relates to the act's stipulations against such damaging mining methods.
**Supporting Evidence:** This is corroborated by the path's coherence and explicit validation regarding legal violations relevant to ecological projects.

## Path 2: Ocean Reclamation Trust -> The Neo-Florida Keys Reef is a major project funde... -> Neo-Florida Keys Reef
**Step-by-Step Explanation:**
1. **Funding Origin:** Settlement fines from Delta-7's violation, manifested in The Ceruleum Affair, resulted in fines directed to the Ocean Reclamation Trust.
2. **Project Support:** The funding from these fines was utilized to support the Neo-Florida Keys Reef, making it an indirect beneficiary.
3. **Outcome:** This project became the largest funded by the settlement, exemplifying the indirect consequence of the legal breach.
**Supporting Evidence:** Document chunks relating to the settlement process and project funding directly support this association. 

## Path Integration Summary

The paths collectively illustrate how the ecological damage and subsequent legal repercussions led to funding an ecological restora
tion initiative. While Path 1 identifies the breach of environmental laws by Delta-7 Mining, Path 2 explains how the remediation fu
nds were employed to support the Neo-Florida Keys Reef project. These paths converge to highlight how legal penalties were redirected towards ecological betterment, closing the loop from violation to restoration.

### References

_No supporting references were provided for inclusion._
```
### 3.6. LightRAG+Reasoning Answer 
```markdown
# Answer
The Neo-Florida Keys Reef project.

# Detailed Reasoning

The question pertains to identifying an ecological project funded as a consequence of an organization violating a conservation act.
 An analysis of the validated path shows a sequence of events stemming from the violation of the Arid Basin Conservation Act of 2060 by Delta-7 Mining and concluding with the funding of the Neo-Florida Keys Reef project.

# Reasoning Path Analysis

## Path 1: Arid Basin Conservation Act of 2060 -> Delta-7 Mining's sonar-pulse technique for mining was banned under the Arid Basin Conservation Act o... -> Delta-7 Mining -> The Veridian Pact subcontracted mineral extraction to Delta-7 Mining to reduce costs, leading to und... -> The Veridian Pact -> The Neo-Florida Keys Reef project was funded by settlement fines from The Ceruleum Affair, benefitin... -> Ocean Reclamation Trust (ORT) -> Neo-Florida Keys Reef
**Step-by-Step Explanation:**
1. **Starting Point:** The Arid Basin Conservation Act of 2060 bans Delta-7 Mining's sonar-pulse mining technique, an ecologically harmful process.
2. **Violation:** Delta-7 Mining continues to use the banned technique, subcontracted by The Veridian Pact for cost reduction.     
3. **Scandal:** The illicit operations lead to The Ceruleum Affair scandal after faults are uncovered by Chief Inspector Maya Petrova during the Brownout of 2088.
4. **Outcome:** Settlement fines from The Ceruleum Affair fund the Ocean Reclamation Trust (ORT).
5. **Project Funding:** ORT channels this funding into a major coral reef rebuilding initiative, the Neo-Florida Keys Reef.        
**Supporting Evidence:** The path outlines a direct link from the conservation act violation to the funding of the Neo-Florida Keys Reef through scandal-induced fines benefiting ORT.

## Path Integration Summary

This sole validated path connects the violation of a conservation act, subsequent organizational scandal, and the funding of an eco
logical project. Delta-7 Mining's improper mining methods, banned under the Arid Basin Conservation Act, set off a chain of repercussions that ultimately financed the Neo-Florida Keys Reef project through settlement fines from The Ceruleum Affair.

# References
No references are listed as specific document chunks were not tagged with reference ids.
```
### 3.8. LightRAG+Reflection Answer 
```markdown
# Answer
The Neo-Florida Keys Reef was the ecological project ultimately funded as a consequence of the Kaito Foundation's subsidiary, Delta-7 Mining, violating the Arid Basin Conservation Act of 2060.

# Detailed Reasoning

The question aims to identify an ecological project directly funded as a result of violations related to material procurement for a
 power source in Centennial City. The key components involve the illicit activities of Delta-7 Mining, the use of Ceruleum for the 
Thorium-Fission Reactor in Centennial City, and the ecological project funded as a result of the fallouts from these activities, specifically under the scandal known as "The Ceruleum Affair."

# Reasoning Path Analysis

There are no validated paths provided that explicitly show the connections among these entities. However, insights can be drawn from the document chunk due to the lack of validated paths.

**Document Chunk Explanation:**
1. **Background Information:** Centennial City is powered by a Thorium-Fission Reactor utilizing Ceruleum, a material involved in mining operations. The reactor's issues led to an investigation into these operations.
2. **Mining Technique Violation:** Delta-7 Mining, a subsidiary of Kaito Foundation, used sonar-pulse mining methods, violating the Arid Basin Conservation Act of 2060 while extracting materials for Ceruleum.
3. **The Ceruleum Affair:** This investigation, led by Chief Inspector Maya Petrova, exposed these violations, leading to fines against involved parties.
4. **Resulting Action:** The resulting fines were allocated to fund the Neo-Florida Keys Reef, which directly supports the mission of the Ocean Reclamation Trust.

**Supporting Evidence:** Although there is no direct evidence via validated paths, the document chunk succinctly describes the chai
n of events from ecological harm due to banned mining techniques, leading to investigations, and finally resulting in fines allocated to the ecological project—the Neo-Florida Keys Reef.

# Reasoning Path Integration Summary

Even though no direct relational paths have been validated in the data provided, the document chunk offers a clear narrative connec
ting the violation of conservation laws by Delta-7 Mining, the subsequent investigation revealing this scandal ("The Ceruleum Affai
r"), and the funding of the Neo-Florida Keys Reef from the resultant fines. Thus, the document supports the notion of this project's indirect funding due to illicit acts related to Centennial City's energy material procurement.

# References

There are no reference documents listed with associated reference_ids for citation.
```