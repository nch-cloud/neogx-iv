import pyhpo
from pyhpo import Ontology
import os
import json

Ontology(data_folder='../assets/hpo-v2025-10-22',transitive=True)
HPO = Ontology
root = HPO.get_hpo_object('HP:0000118')


def get_boundary_term_ids(hpo_ids):
    return set(
        x.id for x in pyhpo.HPOSet([
            HPO.get_hpo_object(xid)
            for xid in hpo_ids
        ]).child_nodes()
    )

def get_boundary_terms(hpo_ids):
    return pyhpo.HPOSet([
            HPO.get_hpo_object(xid)
            for xid in hpo_ids
        ]).child_nodes()

def get_descendant_term_ids(hpo_ids):
    if type(hpo_ids) is str:
        hpo_ids = {hpo_ids}
    desc_ids = set(hpo_ids)
    for xid in hpo_ids:
        desc_ids |= get_descendant_term_ids([y.id for y in HPO.get_hpo_object(xid).children])
    return desc_ids

def get_descendant_terms(hpo_ids):
    if type(hpo_ids) is str:
        hpo_terms = {HPO.get_hpo_object(xid) for xid in hpo_ids}
    desc_terms = set(hpo_terms)
    for x in hpo_terms:
        desc_ids |= get_descendant_terms([y.id for y in xid.children])
    return desc_terms