libname lib '/folders/myfolders/';

/*Chargement des données*/
PROC IMPORT OUT= lib.fichier_donnees 
            DATAFILE= "/folders/myfolders/donnees_qualiy/xAPI-Edu-Data.csv"  
            DBMS=CSV REPLACE;
     		GETNAMES=YES;  * la premiere ligne contient le nom des variables;
     		GUESSINGROWS=MAX; 
RUN;
PROC IMPORT OUT= lib.fichier_donnees_transform
            DATAFILE= "/folders/myfolders/donnees_qualiy/data_transformed.csv"  
            DBMS=CSV REPLACE;
     		GETNAMES=YES;  * la premiere ligne contient le nom des variables;
     		GUESSINGROWS=MAX; 
RUN;
/*Caractérstiques des données*/
proc contents data=lib.fichier_donnees;
/*Histogramme et étude uni pour les varaibles contiues*/
proc univariate data = lib.fichier_donnees; histogram;
run;
/* Étude unidimensionnelle des vraibles qualitatifs*/
proc freq data=lib.fichier_donnees;
table _ALL_/ 
      plots=(freqplot(twoway=cluster orient=v scale=grouppercent)
      	mosaicplot(colorstat=stdres));
run;
/* Corrélation des variables unidimenssionnelle*/
proc corr data =lib.fichier_donnees cov; run;
/* Étude bidimenssionelle et test du khi2*/
proc freq data = lib.fichier_donnees;
table class*_ALL_/ expected cellchi2  chisq ; *CHISQ FAIT LES TESTS;
run;
/*Plots pou étude bidimensionelle*/;
ods graphics on;
proc freq data = lib.fichier_donnees;
table class*_ALL_/ 
      plots=(freqplot(twoway=cluster orient=v scale=grouppercent)
      	mosaicplot(colorstat=stdres));
run;
ods graphics off;

*ACM:;
/*ACM sur les var quantitatives en dim=2 pour visualisation*/
ods graphics on;
proc corresp data = lib.fichier_donnees_transform mca dim=2 outc=sortie ;
tables _ALL_;
supplementary class;
run;
ods graphics off;

*ACM sur les var quantitatives avec l'ensemble des dimensions ; 

proc corresp data = lib.fichier_donnees_transform binary dim=43 outc=sortie noprint ;
tables _ALL_;
supplementary class;
run;

*SORTIE CONTIENT  DES INFORMATIONS AUTRES QUE SUR LES INDIVIDUS ON LES SUPPRIME ET ON GARDE LES COORDONNEES DE 1 à NOMBRE_TOT_DE_DIM  (ICI=24);
data resu_acm; set sortie;
keep dim1-dim43;
run;

*IL FAUT AJOUTER LA var_qual_a_expliquer POUR AVOIR LE FICHIER ENTREE DE LA DISCRIM;
data cible;
set lib.fichier_donnees_transform;
keep class;
run;

data fich_tot;
merge cible resu_acm;
run;
/* Centrage et réduction*/

PROC standard data=fich_tot mean=0 std=1 out=fich_tot1;
	var dim1-dim43;

ods graphics on;
proc cluster data=fich_tot method=ward out=ARBRE PRINT=5;
var DIM1-DIM43;
run;


*TRACE LE DENDROGRAMME ON MET COPY LES DIM POUR POUVOIR FAIRE UNE REPRESENTATION GRAPHIQUE DES CLASSES SUR LE PLAN FACTORIEL;
proc tree data=ARBRE out=PLAN nclusters=3 lineprinter;
COPY DIM1-DIM2;
run;



proc sgplot data=plan;
   scatter y=dim2 x=dim1 / group=cluster;
run;
ods graphics off;

/*SELECTION DE VARIABLES*/
PROC STEPDISC DATA= FICH_TOT  sw;
class class;
var dim1-dim43;
run;

/*--------------------------------------------------------------------------------*/
/*Analyse discriminante sur toutes les variables*/
/* run canonical discriminant factor analysis */
proc candisc data=FICH_TOT ncan=2 out=outcan;
class class;
var dim1-dim43; */
run;



/* scatter plot template */
proc template;
   define statgraph scatter;
      begingraph / attrpriority=none;
         entrytitle 'Class des élèves';
         layout overlayequated / equatetype=fit
            xaxisopts=(label='Canonical Variable 1')
            yaxisopts=(label='Canonical Variable 2');
            scatterplot x=Can1 y=Can2 / group=class name='Class'
                                        markerattrs=(size=6px);
            layout gridded / autoalign=(topright);
               discretelegend 'Class' / border=false opaque=false;
            endlayout;
         endlayout;
      endgraph;
   end;
run;


/* draw scatter plot of individuals in canonical variables 1 and 2 */
ods graphics on / width=10in height=8in;
*/ods listing gpath="/folders/myfolders/image_out";
	proc sgrender data=outcan template=scatter;
	run;
ods graphics off;


* DISCRIMINATION BAYESIENNE ON CALCULE LE % DE MAL CLASSES PAR VALIDATION CROIS2E;
PROC DISCRIM DATA= FICH_TOT  all crossvalidate;
class class;
var dim1-dim43 ;
run;

* DISCRIMINATION BAYESIENNE avec le KPP; 
proc discrim data=FICH_TOT method = npar k=4 crossvalidate ; 
var dim1-dim43 ; 
class class;
run; 

* DISCRIMINATION BAYESIENNE avec boule de rayon R;
proc discrim data=FICH_TOT method = npar R=2 crossvalidate ; 
var dim1-dim43; 
class class;
run;






/*--------------------------------------------------------------------------------*/
/*Analyse discriminante sur les variables séléctionnées(var = dim)*/

/* run canonical discriminant factor analysis */
proc candisc data=FICH_TOT ncan=3 out=outcan;
class class;
var dim1-dim43; */
run;



/* scatter plot template */
proc template;
   define statgraph scatter;
      begingraph / attrpriority=none;
         entrytitle 'Class des élèves';
         layout overlayequated / equatetype=fit
            xaxisopts=(label='Canonical Variable 1')
            yaxisopts=(label='Canonical Variable 2');
            scatterplot x=Can1 y=Can2 / group=class name='Class'
                                        markerattrs=(size=6px);
            layout gridded / autoalign=(topright);
               discretelegend 'Class' / border=false opaque=false;
            endlayout;
         endlayout;
      endgraph;
   end;
run;


/* draw scatter plot of individuals in canonical variables 1 and 2 */
ods graphics on / width=10in height=8in;
*/ods listing gpath="/folders/myfolders/image_out";
	proc sgrender data=outcan template=scatter;
	run;
ods graphics off;


* DISCRIMINATION BAYESIENNE ON CALCULE LE % DE MAL CLASSES PAR VALIDATION CROIS2E;
PROC DISCRIM DATA= FICH_TOT  all crossvalidate;
class class;
var dim1-dim43 ;
run;

* DISCRIMINATION BAYESIENNE avec le KPP; 
proc discrim data=FICH_TOT method = npar k=4 crossvalidate ; 
var dim1-dim43 ; 
class class;
run; 

* DISCRIMINATION BAYESIENNE avec boule de rayon R;
proc discrim data=FICH_TOT method = npar R=2 crossvalidate ; 
var dim1-dim43; 
class class;
run;

;