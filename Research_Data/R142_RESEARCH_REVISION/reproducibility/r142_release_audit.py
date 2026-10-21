"""Verify scientific artifacts and write current claim and response maps."""
from pathlib import Path
import csv, hashlib, importlib.metadata, json, re, sys
ROOT = Path(__file__).resolve().parents[1]
AUD = ROOT/'audit'
RUNS = [f'{family}_batch{i}' for family in ['qwen', 'smollm'] for i in range(1, 5)]
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
def main():
    protocol = json.loads((ROOT/'R142_COMPATIBILITY_PROTOCOL.json').read_text())
    runner_hash = digest(ROOT/'reproducibility/r142_compatibility_experiment.py')
    diagnostic_hash = digest(ROOT/'reproducibility/r142_batching_diagnostic.py')
    assert runner_hash == protocol['runner_sha256']
    recovered = []; generation_records = 0
    for run in RUNS + ['batching_diagnostic']:
        report = json.loads((AUD/f'{run}_provider_log.recovery.json').read_text())
        folder = ROOT/'results'/('numerical_diagnostic/qwen_batch1' if run == 'batching_diagnostic' else f'compatibility/{run}')
        assert len(report) == (10 if run == 'batching_diagnostic' else 25)
        for entry in report:
            path = folder/entry['name']
            assert path.is_file() and entry['status'] == 'verified'
            assert path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256'], path
            recovered.append({'path': str(path.relative_to(ROOT)), 'sha256': entry['sha256'], 'bytes': entry['bytes']})
        saved = json.loads((folder/'protocol.json').read_text())
        assert saved['script_sha256'] == (diagnostic_hash if run == 'batching_diagnostic' else runner_hash)
        if run != 'batching_diagnostic':
            assert saved['spec'] == protocol['batches'][run]
            generation_records += sum(len(json.loads(p.read_text())) for p in (folder/'rows').glob('*.json'))
    assert len(recovered) == 210 and generation_records == 6400
    new_weights = list((ROOT/'results/compatibility').glob('*/adapters/*/adapter_model.safetensors'))
    assert len(new_weights) == 16
    terminal = json.loads((AUD/'R142_compute_terminal_status.json').read_text())
    assert len(terminal['jobs']) == 9 and all(x['status'] == 'COMPLETED' for x in terminal['jobs'])
    deployment = json.loads((ROOT/'results/deployment/SUMMARY.json').read_text())
    assert deployment['script_sha256'] == digest(ROOT/'reproducibility/r142_deployment_simulations.py')
    assert deployment['tape_cases'] == 24562 and deployment['optimization_cases'] == 600
    assert deployment['max_optimization_gap'] == 0 and deployment['workflow_equality_failures'] == 0
    diagnostic = json.loads((ROOT/'results/numerical_diagnostic/qwen_batch1/RESULT.json').read_text())
    assert [diagnostic[k] for k in ['all_first_losses', 'protected_only_losses', 'all_repeat_losses']] == [1, 3, 1]
    assert diagnostic['within_all_text_differences'] == []
    claims = list(csv.DictReader((AUD/'CURRENT_CLAIMS_AND_SOURCES.csv').open()))
    for row in claims: row['evidence_status'] = 'Retained original record. ' + row['evidence_status']
    def claim(key, text, source, scope):
        claims.append({'id': key, 'current_claim': text, 'source': source, 'evidence_status': scope})
    summary = json.loads((ROOT/'results/compatibility_analysis/SUMMARY.json').read_text())
    for name, data in summary['methods'].items():
        claim('R142_' + name,
              f"Mean protected-family loss {data['mean_old_family_loss']:.12f}; mean all-old accuracy {data['mean_all_old_accuracy']:.12f}; mean new accuracy {data['mean_new_accuracy']:.12f}; benefit passes {data['audit_benefit_passes']}/8",
              'results/compatibility_analysis/methods.csv; results/compatibility/*/rows/*.json',
              'All eight batches; full old pools jointly reevaluated; reused tasks; unweighted batch means; not independent deployment populations.')
    claim('R142_paired_training_contrasts', 'MUSCLE vs full: lower/equal/higher old-family loss 6/0/2; vs extra CE 5/1/2', 'results/compatibility_analysis/SUMMARY.json', 'Every paired batch retained; no universal dominance.')
    claim('R142_batch_composition', 'Same weights/device: protected losses 1/72, 3/72, 1/72; full-pool repetition token-identical', 'results/numerical_diagnostic/qwen_batch1/RESULT.json; results/numerical_diagnostic/qwen_batch1/rows/*.json', 'One case; kernel mechanism and other discrepancies not identified.')
    claim('R142_stopping', '24,562 binary-record checks reproduce complete threshold decisions', 'results/deployment/exhaustive_tape_checks.csv; manuscript/R142_DEPLOYMENT_THEORY.tex', 'Fixed bounded record; no population assurance.')
    claim('R142_cut', 'All 600 minimum-cut costs equal exhaustive optima', 'results/deployment/exhaustive_cut_checks.csv; results/deployment/cut_instances.json', 'Known sound-certificate prices; shared deterministic work; classical reduction.')
    claim('R142_workflows', 'All 360 workflows reproduce decisions and artifacts; mean reads 1762.7 versus 2480', 'results/deployment/workflow_replays.csv; results/deployment/workflow_tapes.json', 'Synthetic laws, prices and dependencies; no enterprise savings measured.')
    write_csv(AUD/'R142_CLAIMS_AND_SOURCES.csv', claims)
    text = (ROOT/'manuscript/R142_RESPONSE_TO_COMMENTS.tex').read_text()
    headings = list(re.finditer(r'\\subsection\*\{((?:R|D|F|A)\d+):\s*([^{}]+)\}', text))
    responses = []
    for i, heading in enumerate(headings):
        block = text[heading.end():headings[i+1].start() if i+1 < len(headings) else len(text)]
        block = block.split(r'\section*{')[0].split(r'\end{document}')[0].strip()
        responses.append({'sequence': i+1, 'item': ('latest_' if i < 5 else 'prior_') + heading[1], 'request': heading[2], 'response_tex': block, 'source': 'manuscript/R142_RESPONSE_TO_COMMENTS.tex'})
    assert len(responses) == 47
    write_csv(AUD/'R142_RESPONSE_MATRIX.csv', responses)
    write_csv(AUD/'R142_RECOVERED_ARTIFACTS.csv', recovered)
    preserved = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted((ROOT/'preservation').rglob('*')) if p.is_file()]
    write_csv(AUD/'R142_PRESERVATION_MANIFEST.csv', preserved)
    versions = {name: importlib.metadata.version(name) for name in ['numpy','scipy','matplotlib','PyMuPDF','Pillow']}
    result = {'recovered_artifacts': len(recovered), 'new_adapter_weights': len(new_weights), 'generated_record_evaluations': generation_records, 'protocol_hash_matches_executed_runner': True, 'per_batch_executed_sources_verified': 9, 'completed_jobs': 9, 'deployment_verification': deployment, 'same_weight_diagnostic': diagnostic, 'response_items': len(responses), 'claims': len(claims), 'preserved_files': len(preserved), 'analysis_python': sys.version, 'analysis_dependencies': versions, 'executed_compatibility_sha256': runner_hash, 'executed_diagnostic_sha256': diagnostic_hash}
    (AUD/'R142_RELEASE_AUDIT.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ['recovered_artifacts','new_adapter_weights','generated_record_evaluations','completed_jobs','response_items','claims']}, indent=2))
if __name__ == '__main__': main()
