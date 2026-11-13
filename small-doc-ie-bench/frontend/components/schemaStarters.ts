import type { DynamicSchemaSpec } from "@/lib/api";

// Mirrors adbi_platform document-parsing/schemas/*.schema.json field for field.
export const RESUME_SCHEMA: DynamicSchemaSpec = {
  document_type: "adbi_resume",
  fields: [
    { name: "name", type: "string" },
    { name: "title", type: "string", description: "Intitule de poste affiche en tete du CV, pas l'intitule d'une mission particuliere" },
    { name: "years_experience", type: "number", description: "Nombre total d'annees d'experience professionnelle, en chiffres" },
    { name: "contact", type: "object", fields: [
      { name: "email", type: "string" },
      { name: "phone", type: "string" },
      { name: "linkedin", type: "string" },
      { name: "github", type: "string" },
      { name: "location", type: "string", description: "Ville de residence du candidat, pas le lieu d'une mission" },
    ] },
    { name: "experience", type: "list", fields: [
      { name: "company", type: "string", description: "Employeur qui salarie ou porte le consultant sur cette mission, pas le client final" },
      { name: "end_client", type: "string", description: "Client final chez qui la mission a ete realisee, quand le CV le nomme, distinct de l'employeur" },
      { name: "via", type: "string", description: "Societe intermediaire par laquelle la mission a ete realisee, quand le CV en nomme une, distincte du client final et de l'employeur" },
      { name: "title", type: "string", description: "Role tenu sur cette mission, pas l'intitule de poste affiche en tete du CV" },
      { name: "start_date", type: "date", description: "Date de debut de cette mission" },
      { name: "end_date", type: "date", description: "Date de fin de cette mission, vide si la mission est en cours" },
      { name: "location", type: "string", description: "Lieu ou cette mission a ete realisee, pas la ville de residence du candidat" },
      { name: "context", type: "string", description: "Contexte de la mission en une ou deux phrases (projet, enjeu, client), sans les realisations ni la stack" },
      { name: "achievements", type: "list", description: "Realisations ou taches de la mission, une par entree, recopiees telles qu'ecrites : une puce du CV = une entree", fields: [
        { name: "achievement", type: "string" },
      ] },
      { name: "env_technique", type: "string", description: "Technologies et outils de la mission, separes par des virgules, sans phrase" },
    ] },
    { name: "education", type: "list", fields: [
      { name: "degree", type: "string" },
      { name: "institution", type: "string" },
      { name: "year", type: "string" },
    ] },
    { name: "skills", type: "list", fields: [
      { name: "category", type: "string" },
      { name: "items", type: "list", fields: [
        { name: "item", type: "string" },
      ] },
    ] },
    { name: "languages", type: "list", fields: [
      { name: "language", type: "string" },
      { name: "level", type: "string", description: "Niveau declare pour cette langue, recopie tel qu'imprime" },
    ] },
    { name: "certifications", type: "list", fields: [
      { name: "name", type: "string" },
      { name: "issuer", type: "string" },
      { name: "year", type: "string" },
    ] },
    { name: "interests", type: "list", fields: [
      { name: "interest", type: "string" },
    ] },
  ],
};

export const RECU_SCHEMA: DynamicSchemaSpec = {
  document_type: "adbi_recu",
  fields: [
    { name: "commercant", type: "string", description: "Nom du commerce ou de l'enseigne, en tete du ticket ou de la facture" },
    { name: "date", type: "date", description: "Date d'achat imprimee sur le ticket ou la facture" },
    { name: "tva", type: "list", description: "Lignes du tableau de TVA, une entree par taux imprime", fields: [
      { name: "taux", type: "number", description: "Taux de TVA en pourcentage, par exemple 20 ou 5.5" },
      { name: "base", type: "number", description: "Montant hors taxe imprime pour ce taux" },
      { name: "montant", type: "number", description: "Montant de TVA imprime pour ce taux" },
    ] },
    { name: "total_ht", type: "number", description: "Total hors taxe imprime sur le document" },
    { name: "total_ttc", type: "number", description: "Total toutes taxes comprises a payer, imprime sur le document" },
    { name: "devise", type: "string", description: "Devise des montants, code ISO a trois lettres comme EUR" },
    { name: "moyen_paiement", type: "string", description: "Moyen de paiement imprime : carte, especes, cheque, virement ou autre" },
    { name: "categorie", type: "string", description: "Categorie de la depense, une valeur parmi : repas, transport, hebergement, carburant, fournitures, autre" },
  ],
};

export const FACTURE_SCHEMA: DynamicSchemaSpec = {
  document_type: "adbi_facture",
  fields: [
    { name: "fournisseur", type: "string", description: "Raison sociale du fournisseur qui emet la facture, en en-tete ou en pied de page" },
    { name: "siren", type: "string", description: "Numero SIREN du fournisseur a 9 chiffres, chiffres seuls" },
    { name: "siret", type: "string", description: "Numero SIRET du fournisseur a 14 chiffres, chiffres seuls" },
    { name: "tva_intracom", type: "string", description: "Numero de TVA intracommunautaire du fournisseur, tel qu'imprime" },
    { name: "numero", type: "string", description: "Numero de la facture tel qu'imprime" },
    { name: "date_facture", type: "date", description: "Date d'emission de la facture" },
    { name: "date_echeance", type: "date", description: "Date d'echeance ou date limite de paiement imprimee" },
    { name: "lignes", type: "list", description: "Lignes du tableau de detail de la facture, une entree par ligne imprimee", fields: [
      { name: "designation", type: "string", description: "Libelle de la ligne tel qu'imprime" },
      { name: "quantite", type: "number", description: "Quantite imprimee sur la ligne" },
      { name: "pu_ht", type: "number", description: "Prix unitaire hors taxes imprime sur la ligne" },
      { name: "taux_tva", type: "number", description: "Taux de TVA de la ligne en pourcentage, tel qu'imprime" },
      { name: "total_ht", type: "number", description: "Montant hors taxes imprime sur la ligne" },
    ] },
    { name: "total_ht", type: "number", description: "Montant total hors taxes imprime dans le bloc des totaux" },
    { name: "total_tva", type: "number", description: "Montant total de TVA imprime dans le bloc des totaux" },
    { name: "total_ttc", type: "number", description: "Montant toutes taxes comprises imprime dans le bloc des totaux" },
    { name: "devise", type: "string", description: "Devise des montants imprimee sur la facture, par exemple EUR" },
    { name: "iban", type: "string", description: "IBAN du fournisseur imprime dans les coordonnees bancaires" },
    { name: "bic", type: "string", description: "Code BIC (SWIFT) imprime dans les coordonnees bancaires" },
  ],
};

export const TYPE_PIECE_SCHEMA: DynamicSchemaSpec = {
  document_type: "adbi_type_piece",
  fields: [
    { name: "type_piece", type: "string", description: "Nature du document, une seule valeur parmi : cni (carte d'identite ou passeport), rib, carte_vitale, justificatif_domicile, diplome, autre" },
  ],
};

export const CARTE_VITALE_SCHEMA: DynamicSchemaSpec = {
  document_type: "adbi_carte_vitale",
  fields: [
    { name: "nir", type: "string", description: "Numero de securite sociale (NIR) imprime sous le nom : 13 caracteres puis la cle de 2 chiffres" },
    { name: "nom", type: "string", description: "Nom de famille du titulaire tel qu'imprime sur la carte" },
    { name: "prenoms", type: "string", description: "Prenom(s) du titulaire tels qu'imprimes sur la carte" },
  ],
};

export const JUSTIFICATIF_DOMICILE_SCHEMA: DynamicSchemaSpec = {
  document_type: "adbi_justificatif_domicile",
  fields: [
    { name: "emetteur", type: "string", description: "Organisme qui a emis le document (fournisseur d'energie, operateur, bailleur, impots), en en-tete" },
    { name: "titulaire", type: "string", description: "Nom de la personne a qui le document est adresse" },
    { name: "adresse", type: "string", description: "Adresse postale complete du titulaire : numero, voie, code postal, ville" },
    { name: "date_document", type: "date", description: "Date d'emission du document (date de facture, d'avis ou de quittance)" },
  ],
};

export const DIPLOME_SCHEMA: DynamicSchemaSpec = {
  document_type: "adbi_diplome",
  fields: [
    { name: "intitule", type: "string", description: "Intitule du diplome tel qu'imprime (grade, specialite)" },
    { name: "titulaire", type: "string", description: "Nom et prenom(s) du titulaire du diplome" },
    { name: "etablissement", type: "string", description: "Etablissement ou autorite qui delivre le diplome" },
    { name: "date_obtention", type: "date", description: "Date d'obtention ou de delivrance du diplome" },
  ],
};

export const SCHEMA_STARTERS: DynamicSchemaSpec[] = [
  RESUME_SCHEMA, RECU_SCHEMA, FACTURE_SCHEMA, TYPE_PIECE_SCHEMA,
  CARTE_VITALE_SCHEMA, JUSTIFICATIF_DOMICILE_SCHEMA, DIPLOME_SCHEMA,
];
