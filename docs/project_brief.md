# PawLine Project Brief

PawLine is an after-hours veterinary intake and escalation AI agent for a fictional veterinary clinic.

## Core behavior

- PawLine identifies a fictional customer and pet.
- PawLine collects the reason for the call.
- PawLine uses deterministic code for urgency routing.
- PawLine either books an appointment, creates a human handoff, or answers an approved clinic-policy question.
- PawLine never diagnoses, recommends treatment, or invents availability.
- PawLine uses only synthetic data.
- RAG will later be used only for approved clinic policies.
- Live customer data and appointments will come from API tools, not RAG.

## Safety and scope

- The system must not provide medical diagnoses.
- The system must not recommend treatments.
- The system must not fabricate appointment slots or customer details.
- The system must route to human staff or approved policy responses when needed.

## Current phase

This repository contains only the minimal scaffold for the service. The full AI workflow, data integrations, and operational tools are intentionally deferred.
