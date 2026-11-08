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

<a id="foremal"></a>
### 2c. En riktig hotbar, som i Bed Wars (modell v4)

Modellen har en hotbar med nio platser, ser vad som ligger i varje plats och hur många, och väljer själv plats med
en tangent (1–9), som en spelare. Ett högerklick gör det som saken i handen gör: block placeras bara med block i
handen, ett slag gör svärdets skada bara med svärdet i handen.

- Föremålslistan är Hypixel Bed Wars butik (forumguiden "Every Item In Bedwars", ThePrestige 2020, redigerad april
  2024; den officiella wikin gick inte att nå). Block: ull, härdad lera, sprängsäkert glas, end stone, stege,
  plankor, obsidian. Listan är numrerad för gott och växer bara i slutet (`sim/src/items.rs`, `python/items.py`):
  numret är en del av indata. Det som simulatorn inte kan än (stege, hackor, TNT …) finns i listan men delas aldrig
  ut, så att indata aldrig visar något som beter sig fel.
- Varje runda lägger blocken och svärdet på slumpade platser, så modellen måste läsa av hotbaren. Banor börjar med
  blocken i handen, dueller med svärdet; i arenan "gap" måste den själv byta till block, bygga över och byta tillbaka.
- Det som byggs och delas ut är bara shop-block. Banornas golv byggs av ull, lera, end stone, plankor eller obsidian
  (inte glas: simulatorn bygger golv som hela block, och glas trycker inte ut en spelare). Kartterräng (Treenan,
  Bedwars-kartorna) och slabs/trappor finns kvar, som på riktiga Hypixel-kartor.
- Tangenten trycks före klicken i samma tick, som spelet hanterar ett ticks tangenter.
- Utforskningen av tangenten väger en tiondel (`spatial.ENTROPY_WEIGHTS`), och den börjar på ungefär ett tryck på
  tusen tick: med full vikt drev utforskningen slumpbyten till 1 % av alla tick, och i fighter höll den svärdet bara
  23 % av tiden. Vår egen avvägning, inte från en artikel.
- Den låga vikten räckte inte för att hålla utforskningen vid liv: vid 76M beslut tryckte auto8 en tangent med under
  0,0002 i sannolikhet per tick. Den hittade aldrig att det lönar sig att ta fram blocken, handrundan låg på 0, och
  duell-arenan "gap" stoppade duell-läroplanen i 40M beslut. Därför ett golv under träning (`--hotbar-floor`,
  `spatial.WeightedMultiCategorical.floor`): en andel av tangentvalen dras jämnt fördelat, oavsett hur lågt
  modellens egen sannolikhet går. 0,005 ger ett tryck ungefär var 11:e sekund. PPO räknar med den blandade
  fördelningen, så sannolikheterna stämmer. Spelet, viewern, benchmark och rating spelar utan golv.
- En äldre modell laddas med de nya indata på 0 och de gamla valen oförändrade (`train_auto.transfer`). De två gamla
  värdena för handen ("slot 0", "slot 1") betyder nu "block eller tom hand" och "svärd i handen": samma värden som
  förut i varje gammal situation.

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
- När modellen tar över 0,85 i snitt mot referenserna läggs en fryst kopia av den till, med modellens Elo just då
  (`rating.ADD_AT`). Annars säger en rating bara "mycket bättre": i auto7 svängde Elo 1350–1660 när den slog nästan
  alla. Ankaret och de tre nyaste kopiorna behålls. Självspelets poängtal till poolen räknas bara från 1 mot 1:
  mot två förlorar den de flesta rundor oavsett vilken version som spelar den första.

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
som räknas är vinst och skada (en crit ger redan 1,5× skada). Deras vikt w är den andel av duell-läroplanen som
återstår: 1 på första nivån, 0 när alla duell-rattar är i botten (`train_auto.bonus_weight_for`). Den syns aldrig i
indata.

Förr följde w vinsten mot modellens frysta kopior. Från noll föll den för tidigt: vid 14M slog auto8 sina 2M- och
6M-kopior i 90 % eftersom de var nybörjare, och w föll till 0,03 med duell-läroplanen på 9 av 20. De punkter nedan
som gäller den gamla regeln är historik.

- Källa: OpenAI Fives ablation med bara vinst/förlust nådde TrueSkill 155 mot skriptade bottens 100, men med "a
  large penalty to sample efficiency": stöttning köper fart i början och behövs inte för slutlig styrka.
- Formeln är vår (extrapolation): w = clip(1 − (vinst mot referenser − 0,5) / 0,4, 0, 1). Närmast i litteraturen
  är OpenAI Fives kompetensstyrda införande av nya funktioner.
- Bara de frysta kopiorna av modellen räknas i vinstandelen, inte de skriptade bottarna: dem slår modellen redan
  ~90 %, så de skulle sänka w direkt utan att säga något om just dessa tekniker.
- Kod: `sim/src/shaping.rs` (beräkningen, samma i simulatorn och spelet), `train_auto.bonus_weight_for` (w),
  `Fight18.players()` (motståndarnas slag och träffar i riktiga spelet).
- Vi avstår från idén "belöna crit när den inte hänt på länge": en sådan utlösare gör belöningen
  icke-monoton (bonusen kommer tillbaka när beteendet blir ovanligt) och kan få modellen att pendla.

<a id="slag-mot-ingen"></a>
### Slag mot ingen

Ett vänsterklick utan någon motståndare i synfältet kostar 0,01 (`shaping::BLIND_SWING_COST`, delen
"swing at nobody"). Synfältet är 60° åt varje håll från dit den tittar, ungefär spelets bredd vid FOV 90. I en
bana finns ingen att slå, så varje klick kostar. Kostnaden viktas inte ned och är inte nollsumma: den är ingen
stöttning som ska bort, och ingen motståndare kan få modellen att betala den. Indata ändras inte: modellen ser
redan sin egen blickriktning och motståndarnas lägen. Önskat av projektägaren 2026-11-07.

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
- Så som det är byggt (`python/pool.py`): en ny version var 5 miljoner beslut, högst 30 (den äldsta går, utom
  den första). Av självspelsrundorna 80 % mot senaste, resten pool via (1−x)² med 10 % jämnt golv; x är modellens
  poäng mot versionen, där nyare resultat väger mer (×0,98 per runda). Skriptade botar: hälften av träningsduellerna
  så länge duellkursen inte är klar, sedan 10 %. Undantaget från "upp till 10 %" är medvetet: kursen höjs bara av
  botrundor, och självspelsrundor räknas aldrig in i den eller i provens resultat. Proven spelas bara mot botar.

<a id="svagare"></a>
### 8. Svagare motståndare = äldre versioner, inte slumpdrag

Behövs en svagare motståndare tas en äldre version. En nyare version som ibland gör slumpdrag spelar inte som en
svagare spelare utan som en expert som ibland får kramp, och det lär modellen att straffa slumpmässiga misstag i
stället för att slå en bra motståndare. Det är inte testat i litteraturen åt något håll; äldre versioner är det
beprövade sättet.

## Att veta att den blir bättre

<a id="mata"></a>
### Mätning med felmarginal

Måtten vi styrde efter var till stor del brus (auto7, november 2026): reglagen rörde sig 96 gånger upp och 89 ned,
132 av dem `hop.longer` och `hop.bend` fram och tillbaka; proven (32 rundor) svängde lika mycket som slumpen själv
(±0,08); Elo svängde 300 poäng mellan ratingar. Därför:

- **Fast prov** (`python/benchmark.py`): samma rundor varje gång (fasta seeds), tre fasta svårigheter per sort och en
  runda där hon måste hitta blocken, 128 rundor var som räknas en gång, både med slump och utan, med 95 %-intervall
  (Wilson). Var 10:e miljon beslut till `benchmark.jsonl`. Proven på aktuell nivå blir svårare när kursen går
  framåt, så de visar inte framsteg över tid; det fasta provet gör det.
- **Färdigheter som poängen inte visar** (`Env.Skills`): slag per sekund, missar, crits, långa slag, svärdsträffar,
  och hur ofta och hur snabbt den hittar blocken. Räknas bara, syns aldrig i indata och ger inga poäng.
- **Kursen beslutar med marginal:** ett reglage höjs bara när vinstandelen säkert är över 0,6 och sänks bara när den
  säkert är under 0,35 (95 % ensidigt, Wilson, upp till 60 rundor), i stället för 70 % på 30 rundor.
- **Elo ur alla matcher:** varje rating spelar varje referens i tre sorters dueller, 64 rundor var, och alla
  matcher sparas. Alla ratingar anpassas tillsammans (Bradley–Terry, ankaret 1000, en svag dragning mot 1000 så
  att ett rent svep inte springer iväg) och får ett intervall. Förr fick varje referens ett fast Elo från en brusig
  rating, och felen staplades.
- Källa för grundprincipen: Agarwal m.fl. 2021, "Deep RL at the Edge of the Statistical Precipice": få rundor och
  inga intervall är den vanligaste orsaken till övertolkade RL-resultat.

<a id="natet"></a>
### Nätverksval som prövas (alla startar exakt som den gamla modellen)

- **Minne i värdet** (`--critic-memory`): värdet hade inget minne och gissade bara utifrån nuet, i strider med dold
  information. Dess nya minne startar som en kopia av policyns, och de första 1M besluten lär sig bara värdet, så
  att dess fel inte puttar policyn.
- **Standardiserade indata** (`--normalize-inputs`): värdena hade mycket olika skala. Medel och spridning mäts en
  gång och vävs in i första lagret, så svaren är oförändrade; spridningar under 0,25 höjs dit, så att ett nästan
  alltid tomt värde inte blåses upp. Andrychowicz m.fl. 2021 ("What Matters in On-Policy RL") fann normalisering
  av indata bland det viktigaste för PPO.
- **Egna nät för motståndare och hotbar** (`--entities`): varje motståndare och varje plats går genom samma lilla
  nät och slås ihop (max och summa), som i OpenAI Five och AlphaStar: det som lärs om en motståndare gäller alla,
  i vilken ordning och hur många de än är. Läggs till genom ett lager som startar på noll.
- Inte prövat nu: bredare nät och separat värdenät; spelandet tar redan 6–7 av 10 s per omgång.
- Resultat efter 3M beslut från smoke_v4: alla skillnader inom felmarginalen (±0,06). Inget av dem löser
  handrundan (0/128).

<a id="fem"></a>
### Lärdomar från OpenAI Five (arXiv 1912.06680)

- **Lika lång tid, inte lika många beslut:** Five fann att varje datapunkt bara ska användas ungefär en gång (2–3
  gånger halverade farten, 8 stoppade inlärningen). Vi kör 4 pass. `--epochs` jämförs med `--minutes`, så att
  färre pass får spela mer under samma tid.
- **En diskonteringsfaktor per körning** (`--gamma`, i net.json): längre horisont hjälpte en redan tränad Five. Vår
  0,995 är ungefär 10 s. Siktets potential (env.rs `Env.gamma`) får alltid samma γ som PPO, annars kan stöttningen
  ändra vad som är bäst.
- **Infasning** (`Progress.phase_in`): en ny regel går från 0 till full andel under N beslut. Slumpad hand påslagen
  på en gång tog provet från 31 till 19 (bridge) och 10 (stairs). Five såg samma sak med buyback.
- **Stilla start efter omformning:** när `transfer()` har ändrat nätet lär den sig med takt 0 i 300k beslut
  (`STILL_AFTER_SURGERY`), så att Adams moment hinner ikapp innan något rör sig.
- **Rating mot jämbördiga:** en bot som slagits ≥ 95 % (eller ≤ 5 %) spelas bara var fjärde rating; frusna kopior
  alltid. Five spelade bara referenser inom 15–85 %.
- **Börja om från noll när koden satt sig:** Fives omkörning med slutlig kod nådde samma nivå på ~20 % av
  beräkningen och vann sedan 98 % mot originalet, som hade fastnat lägre efter 20 omformningar. Ingen inspelad data:
  den nya ska inte ärva den gamlas vanor. Går det för sakta finns kickstarting (Schmitt m.fl. 2018): en dragning
  mot auto7:s val som trappas ner till noll.
- **Kickstarting** (`--kick LÄRARE`, `train_auto.Kickstart`): auto8 från noll hittade aldrig bridge (0 av 60 på
  första nivån efter 10M). Lärarens val sparas för varje tick i rundorna modellen själv spelar, och
  korsentropin mot dem läggs till förlusten med vikt 1, som faller linjärt till 0 vid 20M beslut. Därefter räknas
  bara belöningen. Hotbar-tangenten lärs inte ut: läraren tar aldrig fram sina block, och det är en vana den nya
  inte ska få.

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
