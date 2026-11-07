# Träningens design: varför vi gör som vi gör

Det här dokumentet förklarar de val i träningen som inte följer av spelets regler utan är beslut: hur motståndarna
väljs, hur belöningen är byggd, vad vi mäter och vad vi medvetet inte gör. Varje beslut har en källa eller är
markerat som vår egen extrapolation. Koden hänvisar hit (`docs/training-design.md#...`). Bakgrunden är
forskningsgenomgången `reports/Utforskning och self play för PvP.md` (november 2026).

## Grundregler

<a id="sanna-indata"></a>
### 1. Indata ljuger aldrig

Allt modellen ser beskriver spelet som det är. Önskemål om hur den ska spela ("sikta utifrån, mot bröstet")
ligger i belöningen, aldrig i indata. Exempel: "ett klick nu skulle träffa" följer spelets egen regel, även när
ögonen är inne i motståndarens hitbox. Det betyder också att inga räknare för belöningen (som SOFE gör) får
läggas i observationen: de är inte speltillstånd och finns inte på en riktig server.

<a id="paritet"></a>
### 2. Samma regel i simulatorn och i Python

Varje belöning och varje indata finns både i `sim/src/env.rs` och i `python/auto_env.py`, och
`tests/test_sim_env.py` kontrollerar att de ger exakt samma tal. Riktiga spelet kontrolleras mot simulatorn med
`checks/check_duel_real.py`.

## Mätning

<a id="referenser"></a>
### 3. Frysta referensmotståndare och Elo

Vinstandel mot träningsmotståndarna bevisar inget: den kan stiga medan modellen går i cirklar mellan stilar. Därför
mäts styrkan mot en fast uppsättning motståndare som aldrig tränas mot: de skriptade botarna och utvalda frysta
versioner av modellen. Resultatet blir ett Elo-tal och en vinstmatris.

- Källa: OpenAI Five höll 83 frysta referensagenter och lät bara matcher inom 15–85 % förväntad vinst räknas
  ([arXiv 1912.06680](https://arxiv.org/abs/1912.06680)); DeepMinds fotboll utvärderade mot 13 agenter som aldrig
  användes i träningen ([arXiv 2105.12196](https://arxiv.org/abs/2105.12196)).
- Vårt val (extrapolation): utvärdering på en egen liten uppsättning rundor, så att träningen inte blir långsammare.

## Belöning

<a id="nollsumma"></a>
### 4. Stöttning är nollsumma

En stöttningsterm (sikte, långa slag, crit, missade slag) räknas för båda sidor, och modellen får sin egen minus
motståndarens. Då går den inte att farma: i self-play kan två versioner av modellen annars samarbeta om att samla
bonusar i stället för att vinna. Skada är redan symmetrisk (utdelad minus mottagen).

- Källa: OpenAI Five drog av motståndarnas medelbelöning från varje hjältes belöning; GT Sophy gjorde sin
  passeringsbonus symmetrisk för att det inte skulle finnas "positive-cycle reward loops to exploit"
  ([GT Sophy, Nature 2022](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)).

<a id="potential"></a>
### 5. Sikte som potential

Sikte belönas som skillnaden i en potential, F = γ·Φ(s′) − Φ(s), där Φ = −k·vinkelfel mot motståndarens bröst,
γ = samma som i PPO (0,995) och Φ = 0 när rundan är slut. En sådan term ändrar bevisat inte vilket beteende som är
bäst, bara hur snabbt det hittas, eftersom summan över en runda bara beror på start- och sluttillståndet.
Tidigare gav vi en fast poäng per tick med siktet på motståndaren; den kunde farmas genom att stå och sikta.

- Källa: Ng, Harada & Russell 1999 (garantin); Müller & Kudenko 2025 om hur skalan påverkar nyttan
  ([arXiv 2502.01307](https://arxiv.org/abs/2502.01307)).
- Vi gör inte om rörelsebanornas framstegsbelöning nu: fem banatyper är maxade och stabila, och en respawn
  teleporterar spelaren, vilket skulle ge ett hopp i potentialen. Det får i så fall prövas separat.

<a id="nedtrappning"></a>
### 6. Händelsebonusar trappas ned när modellen kan

Crit, långa slag och kostnaden för missade slag är stöttning: de hjälper modellen att upptäcka tekniker, men det
som räknas är vinst och skada (en crit ger redan 1,5× skada). Deras vikt w sjunker när vinstandelen mot
referenserna stiger, och får bara sjunka (backas om referensstyrkan faller). Vikten sparas mellan omstarter och
syns aldrig i indata.

- Källa: OpenAI Fives ablation med bara vinst/förlust nådde TrueSkill 155 mot skriptade bottens 100, men med "a
  large penalty to sample efficiency": stöttning köper fart i början och behövs inte för slutlig styrka.
- Formeln är vår (extrapolation): w = clip(1 − (vinst mot referenser − 0,5) / 0,4, 0, 1). Närmast i litteraturen
  är OpenAI Fives kompetensstyrda införande av nya funktioner.
- Bara de frysta kopiorna av modellen räknas i vinstandelen, inte de skriptade bottarna: dem slår modellen redan
  ~90 %, så de skulle sänka w direkt utan att säga något om just dessa tekniker.
- Kod: `sim/src/shaping.rs` (beräkningen, samma i simulatorn och spelet), `train_auto.bonus_weight_from` (w),
  `Fight18.players()` (motståndarnas slag och träffar i riktiga spelet).
- Vi avstår från idén "belöna crit när den inte hänt på länge": en sådan utlösare gör belöningen
  icke-monoton (bonusen kommer tillbaka när beteendet blir ovanligt) och kan få modellen att pendla.

## Motståndare

<a id="self-play"></a>
### 7. Self-play mot senaste versionen och en pool av äldre

De flesta dueller spelas mot senaste versionen av modellen, resten mot en pool av sparade äldre versioner, plus en
liten andel skriptade botar. Äldre versioner väljs efter hur ofta modellen förlorar mot dem (PFSP), med ett golv av
jämnt urval så att ingen glöms.

- Källa: OpenAI Five spelade 80 % mot senaste och 20 % mot äldre för att undvika "strategy collapse"; Bansal m.fl.
  fann att enbart senaste versionen gjorde att ena sidan föll efter permanent
  ([arXiv 1710.03748](https://arxiv.org/pdf/1710.03748)); AlphaStar/TStarBot-X viktar med f_hard(x) = (1−x)^p
  ([arXiv 2011.13729](https://arxiv.org/abs/2011.13729)); GT Sophy: "the straightforward application of
  self-play was inadequate" och höll skriptade motståndare med hela träningen.
- Våra siffror (extrapolation): 75–80 % senaste, resten pool via (1−x)² med 10 % jämnt golv, upp till 10 %
  skriptade botar; en ny version i poolen var 2–10 miljoner beslut.

<a id="svagare"></a>
### 8. Svagare motståndare = äldre versioner, inte slumpdrag

Behövs en svagare motståndare tas en äldre version. En nyare version som ibland gör slumpdrag spelar inte som en
svagare spelare utan som en expert som ibland får kramp, och det lär modellen att straffa slumpmässiga misstag i
stället för att slå en bra motståndare. Det är inte testat i litteraturen åt något håll; äldre versioner är det
beprövade sättet.

## Det vi inte gör (nu)

<a id="ingen-e3b"></a>
### 9. Ingen nyfikenhetsbonus (E3B, RND) än

Generella nyfikenhetsbonusar har sitt stöd i glesa navigationspussel (E3B på MiniHack,
[arXiv 2210.05805](https://arxiv.org/abs/2210.05805)). I det enda oberoende testet med tät belöning, Craftax
(ICML 2024, [arXiv 2402.16801](https://arxiv.org/abs/2402.16801)), hjälpte ingen av dem och E3B sänkte resultatet;
RND förbättrar PPO i mindre än hälften av Atari-spelen ([arXiv 2407.13755](https://arxiv.org/html/2407.13755v3)).
Våra belöningar är täta. Det blir aktuellt först med en gles uppgift, till exempel hela matcher där bara vinst
räknas.

<a id="ingen-llm"></a>
### 10. Inga belöningar från språk- eller bildmodeller för strid

MineCLIP var sämre än handskrivna belöningar på strid i Minecraft (Pigman 1,6 % mot 13,6 %,
[arXiv 2206.08853](https://arxiv.org/html/2206.08853)), och de snabba varianterna klarar bara vår hastighet med en
mycket liten belöningsmodell.

## Begränsningar

- Self-play sker i simulatorn. Viewern och riktiga spelet har skriptade botar: en modellstyrd bot där kräver att
  modden skickar block och karta runt varje bot, vilket är ett separat steg.
- Ingen har publicerat något om PvP i Minecraft; siffror märkta "extrapolation" är våra egna och ska prövas.
