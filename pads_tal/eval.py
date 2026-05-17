import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import ast
import gc
# add metrics src to path (stable-audio-metrics root so that 'src' is importable)
metrics_root = os.path.join(os.path.dirname(__file__), 'tools', 'stable-audio-metrics')
if metrics_root not in sys.path:
    sys.path.insert(0, metrics_root)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
from src.passt_kld import passt_kld
from tools.eval_tools import clap_score, _load_model, _get_audio_embed
from torch.utils.data import DataLoader
from tools.ipr import IPR, get_ipr_info
from tools.args_mult import CSVorSpaceList
from tools.diversity_metrics import (
    TruncatedVendi
)
import glob

def clear_gpu_memory():
    """Function to completely clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        # Safe tensor cleanup - prevent weak reference errors
        try:
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and obj.is_cuda:
                        del obj
                except (ReferenceError, RuntimeError):
                    # Ignore weak reference or runtime errors
                    pass
        except Exception:
            # Ignore errors in the entire loop
            pass

        gc.collect()
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"GPU memory status - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

def print_gpu_memory_usage(stage_name):
    """Function to print GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        print(f"[{stage_name}] GPU memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB, Max: {max_allocated:.2f}GB")

def clear_tensorflow_memory():
    """Function to completely clear TensorFlow memory"""
    print("Clearing TensorFlow memory...")
    
    try:
        # Check if TensorFlow is loaded
        import tensorflow as tf
        
        # Clear TensorFlow session
        if hasattr(tf, 'keras'):
            tf.keras.backend.clear_session()
        
        # Clear TensorFlow GPU memory
        if hasattr(tf.config, 'experimental'):
            try:
                # Disable GPU memory growth setting
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.reset_memory_growth(gpu)
            except Exception as e:
                print(f"Error disabling TensorFlow GPU memory settings: {e}")
        
        # Clear TensorFlow memory
        if hasattr(tf, 'reset_default_graph'):
            tf.reset_default_graph()
        
        print("TensorFlow memory cleared")
        
    except ImportError:
        print("TensorFlow is not installed or not loaded")
    except Exception as e:
        print(f"Error while clearing TensorFlow memory: {e}")

def run_clap_eval(generated_path, id2text, ext=".mp3", num_avg=1, clap_basemodel='audio', use_seg=False):
    if use_seg:
        num_avg=1
        
    if clap_basemodel == 'audio':
        clap_model = '630k-audioset-fusion-best.pt'
    elif clap_basemodel == 'music':
        clap_model = 'music_audioset_epoch_15_esc_90.14.pt'
        
    print("\n>>> [Starting CLAP Evaluation]")
    clp_result = []
    clp_scores_dict_all = []
    for i in range(num_avg):
        clap_results = clap_score(id2text, generated_path, audio_files_extension=ext, clap_model=clap_model, use_seg=use_seg)
        if isinstance(clap_results, (list, tuple)) and len(clap_results) > 1:
            clap_i = clap_results[0]
            clp_scores_dict_all.append(clap_results[1])
        else:
            clap_i = clap_results
        clp_result.append(clap_i.cpu() if hasattr(clap_i, 'cpu') else clap_i)
    
    print(f"CLAP Score (for {num_avg} iterations): {np.mean(clp_result):.4f}")
    return clp_result, clp_scores_dict_all

def compute_vendi_score(embeddings):

    Truncated_vendi = TruncatedVendi(embeddings)

    vendi_score = Truncated_vendi.compute_score(
        alpha=1.0,
        kernel="cosine",
        sigma=5.0,
        use_nystrom=False,
        batch_size=128,
    )
    
    return vendi_score

def save_all_files_to_csv(clp_scores_dict_all=None, kl_scores_dict=None, result_path="result"):
    os.makedirs(result_path, exist_ok=True)
    
    # collect file keys
    all_keys = set()
    if clp_scores_dict_all:
        for clp_dict in clp_scores_dict_all:
            if clp_dict:
                all_keys.update(clp_dict.keys())

    if kl_scores_dict:
        all_keys.update(kl_scores_dict.keys())

    
    if not all_keys:
        return
    
    # collect data
    scores_data = []
    for key in sorted(all_keys, key=str):
        row_data = {'id': key}
        
        if kl_scores_dict and key in kl_scores_dict:
            row_data['kl'] = kl_scores_dict[key]
        
        
        if clp_scores_dict_all:
            for i, clp_dict in enumerate(clp_scores_dict_all, 1):
                if clp_dict and key in clp_dict:
                    row_data[f'clap#{i}'] = clp_dict[key]
        scores_data.append(row_data)
    
    if scores_data:
        scores_df = pd.DataFrame(scores_data)
        scores_csv_path = os.path.join(result_path, 'individual_scores.csv')
        scores_df.to_csv(scores_csv_path, index=False)
        print(f"Individual scores saved to {scores_csv_path}")
        print(f"Saved columns: {list(scores_df.columns)}")
    else:
        print("No data to save.")


def main():
    # Check and print GPU settings
    if torch.cuda.is_available():
        print(f"CUDA available - Number of available GPUs: {torch.cuda.device_count()}")
        print(f"Current default GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")
        print(f"CUDA_VISIBLE_DEVICES environment variable: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    else:
        print("CUDA is not available.")
    
    parser = argparse.ArgumentParser(description='Evaluate audio with CLAP / KLpasst / FDopenl3 (songdescriber nosinging).')

    parser.add_argument('--root-path', required=True, help='Root folder for experiment results (wav files exist under each experiment folder)')
    parser.add_argument('--dataset', choices=['songdescriber', 'melbench'], required=True)
    parser.add_argument('--ext', default='.wav', help='Audio file extension')
    parser.add_argument('--result-path', default='', help='result file save to ...')

    parser.add_argument('--modes', action=CSVorSpaceList, choices=['clap', 'kld', 'fd', 'ipr', 'vendi'],  default=['clap', 'kld', 'ipr', 'vendi'])
    parser.add_argument('--text-type', choices=['caption', 'tag'], default='tag', help='Select text type (tag only available)')
    parser.add_argument('--num-avg', type=int, default=10, help='CLAP and IPR average size', required=False)
    parser.add_argument('--clap-basemodel', default="music", choices=['audio', 'music'])
    parser.add_argument('--ipr-basedata', default="songdesc", choices=['songdesc', 'songdescriber', 'melbench'])
    parser.add_argument("--clap-avg", action="store_true", default=False,
                       help="Use 10 times average instead of front/mid/end segment(10s) for CLAP evaluation")
    parser.add_argument('--dataset-genre', type=str, default='', choices=[
            'elec', 'jazz', 'pop','newage', 
            'blues', 'classic', 'country', 'latin', 'metal','rock', 'easy', 'folk', 'rnb', 'hiphop', 'world'], required=False)
    args = parser.parse_args()
    
    args.num_avg = args.num_avg if args.num_avg>=1 else 1
    root_path = args.root_path
    if 'ipr' in args.modes and args.dataset != args.ipr_basedata :
        raise Exception("[Error] Currently dataset and ipr_basedata must be same : TODO")

    # dataset-specific settings
    if  args.dataset == 'songdescriber': 
        id_col = 'caption_id'
        if args.text_type == 'tag':
            text_col = 'aspect_list'
        else: # songdescriber (caption)
            text_col = 'caption'

        csv_file_path = 'input_inf/eval/baseline_songdesc_tags.csv'
        #csv_file_path = 'tools/stable-audio-metrics/load/song_describer-nosinging.csv'

        ref_prob_path = os.path.join(os.path.dirname(__file__), 'tools', 'stable-audio-metrics', 'load', 'passt_kld', 'song_describer-nosinging__collectmean__reference_probabilities.pkl')
        ref_openl3_path = os.path.join(os.path.dirname(__file__), 'tools', 'stable-audio-metrics', 'load', 'openl3_fd', 'song_describer-nosinging__channels2__44100__openl3music__openl3hopsize0.5__batch4.npz')
        no_ids = []
        
    elif args.dataset =='melbench':
        id_col = 'ytid'
        if args.text_type == 'tag':
            text_col = 'aspect_list'
            if args.dataset_genre:
                if args.dataset_genre == 'elec':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_electronic.csv'
                elif args.dataset_genre == 'jazz':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_jazz.csv'
                elif args.dataset_genre == 'pop':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_pop.csv'
                elif args.dataset_genre == 'newage':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_new_age.csv'
                elif args.dataset_genre == 'blues':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_blues.csv'
                elif args.dataset_genre == 'classic':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_classical.csv'
                elif args.dataset_genre == 'country':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_country.csv'
                elif args.dataset_genre == 'latin':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_latin.csv'
                elif args.dataset_genre == 'metal':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_metal.csv'
                elif args.dataset_genre == 'rock':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_rock.csv'
                elif args.dataset_genre == 'easy':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_easy_listening.csv'
                elif args.dataset_genre == 'folk':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_folk_acoustic.csv'
                elif args.dataset_genre == 'rnb':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_rnb.csv'
                elif args.dataset_genre == 'hiphop':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_hip_hop.csv'
                elif args.dataset_genre == 'world':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_world_traditional.csv'
                else:
                    raise Exception("[Error] MelBench dataset for genre {} is not supported".format(args.dataset_genre))
            else:
                csv_file_path = 'input_inf/eval/baseline_melbench.csv'
        else:
            raise Exception("[Error] MelBench dataset for sentences not supported")
        
        ref_prob_path = os.path.join(os.path.dirname(__file__), 'tools', 'stable-audio-metrics', 'load', 'passt_kld', 'song_describer-nosinging__collectmean__reference_probabilities.pkl')
        ref_openl3_path = os.path.join(os.path.dirname(__file__), 'tools', 'stable-audio-metrics', 'load', 'openl3_fd', 'song_describer-nosinging__channels2__44100__openl3music__openl3hopsize0.5__batch4.npz') 
        no_ids = []
    else:  
        raise Exception("")

    results = {'root_path': root_path}
    print("********************************************************")
    print("Let's start with Generated path : \n    {}".format(root_path))
    print("********************************************************")

    # ====================================== CLAP Score
    clp_scores_dict_all = []
    df = pd.read_csv(csv_file_path)
    if 'clap' in args.modes:
        print('\n\nComputing CLAP score..')

        # Initialize GPU memory
        clear_gpu_memory()

        # Process text based on text_type
        if args.text_type == 'tag':
            # Convert tag list to string for CLAP
            df[text_col] = df[text_col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            id2text = df.set_index(id_col)[text_col].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x)).to_dict()
            print(f'Using tags as text input for CLAP (converting list to comma-separated string)')
        else:
            id2text = df.set_index(id_col)[text_col].to_dict()
            print(f'Using {args.text_type} as text input for CLAP')

        # default: CLAP-f
        clps, clp_scores_dict_all = run_clap_eval(root_path, id2text, ext=args.ext, num_avg=args.num_avg, clap_basemodel=args.clap_basemodel, use_seg=not args.clap_avg)
        clp = np.mean(clps)
        
        print(f'[{args.dataset}] CLAP: {clp:.4f}')
        results['clap_score'] = clp
        results['clap_score_all'] = clps
        
        # Delete CLAP related variables
        del clps, clp, id2text
        
        clear_gpu_memory()

    # ====================================== KLD-Passt Score
    kl_scores_dict = {}
    if 'kld' in args.modes :
        print('\n\nComputing KLpasst..')
        # Initialize GPU memory
        clear_gpu_memory()
        
        kl_results = passt_kld(ids=df[id_col].tolist(), 
                    eval_path=root_path, 
                    load_ref_probabilities=ref_prob_path,
                    no_ids=no_ids,
                    eval_files_extension=args.ext,
                    collect='mean')
       
        if isinstance(kl_results, (list, tuple)) and len(kl_results) > 1:
            kl = kl_results[0]
            kl_scores_dict = kl_results[1]
        else:
            kl = kl_results
            kl_scores_dict = {}

        print(f'[{args.dataset}] KLpasst: {kl:.4f}')
        results['klpasst'] = kl
        
        # Delete KLD related variables
        del kl_results, kl

        clear_gpu_memory()
    
    # ====================================== FD-openl3 Score
    if 'fd' in args.modes:
        print('\n\nComputing FDopenl3..')
        # Initialize GPU memory
        clear_gpu_memory()
        
        # Dynamic import of OpenL3 FD module
        from src.openl3_fd import openl3_fd
        
        try:
            fd = openl3_fd(
                channels=2,
                samplingrate=44100,
                content_type='music',
                openl3_hop_size=0.5,
                eval_path=root_path,
                eval_files_extension=args.ext,
                load_ref_embeddings=ref_openl3_path,
                batching=2
            )
            results['fdopenl3'] = fd
            print(f'[{args.dataset}] FDopenl3: {fd:.4f}')
            
        finally:
            # Delete FD related variables
            if 'fd' in locals():
                del fd
            clear_tensorflow_memory()
            
            clear_gpu_memory()
    # ====================================== IPR
    if 'ipr' in args.modes:
        # IPR score
        print('\n\nComputing IPR..')
        # Initialize GPU memory
        clear_gpu_memory()

        ipr_ref_options, ipr_gen_options, ipr_num_avg = get_ipr_info(args.ipr_basedata, args.clap_basemodel, args.num_avg, generation_target=args.dataset, dataset_genre=args.dataset_genre)
        ipr_reference = IPR(50, 3, -1, **ipr_ref_options)

        precision_results=[]
        recall_results=[]
        for a in range(ipr_num_avg):
            precision, recall = ipr_reference(root_path, **ipr_gen_options)
            if ipr_num_avg !=1 :
                print("\t Mid Term Precision: {:.4f}, Recall : {:.4f}".format(precision,recall))
            precision_results.append(precision)
            recall_results.append(recall)
        
        results['precision_all'] = precision_results
        results['precision_mean'] = np.mean(precision_results)
        results['precision_std'] = np.std(precision_results)
        results['recall_all'] = recall_results
        results['recall_mean'] = np.mean(recall_results)
        results['recall_std'] = np.std(recall_results)
        print('[{}] IPR precision : {:.4f}({:.4f})'.format(args.ipr_basedata, results['precision_mean'], results['precision_std']))
        print('[{}] IPR recall :{:.4f}({:.4f})'.format(args.ipr_basedata, results['recall_mean'], results['recall_std']))
        
        # Delete IPR related variables
        del ipr_reference, precision_results, recall_results, precision, recall
        
        # Clean up memory after IPR
        #print_gpu_memory_usage("After IPR")
        clear_gpu_memory()
    
    # ====================================== Vendi Score
    if 'vendi' in args.modes:
        print('\n\nComputing Vendi Score..')
        # Initialize GPU memory
        clear_gpu_memory()
        
        # Load CLAP model (for vendi)
        if args.clap_basemodel == 'audio':
            clap_model = '630k-audioset-fusion-best.pt'
        elif args.clap_basemodel == 'music':
            clap_model = 'music_audioset_epoch_15_esc_90.14.pt'
        
        vendi_model = _load_model(clap_model)
        if torch.cuda.is_available():
            vendi_model = vendi_model.to('cuda')
        vendi_model.eval()
        
        try:
            audio_files = [os.path.join(root_path, fn) for fn in os.listdir(root_path) if fn.lower().endswith(args.ext.lower())]
            
            if not audio_files:
                print(f'[{args.dataset}] Vendi: No audio files found')
                results['vendi'] = None
            else:
                print(f'Processing {len(audio_files)} audio files for Vendi Score...')
                
                embeddings = []
                for file in tqdm(sorted(audio_files), desc="Extracting audio embeddings"):
                    try:
                        emb = _get_audio_embed(file, vendi_model, use_seg=not args.clap_avg)
                        if isinstance(emb, torch.Tensor):
                            emb = emb.flatten()
                        elif isinstance(emb, np.ndarray):
                            emb = torch.from_numpy(emb).flatten()
                        embeddings.append(emb)
                    except Exception as e:
                        print(f"  Warning: Error processing {file}: {e}")
                        continue
                
                if len(embeddings) == 0:
                    print(f'[{args.dataset}] Vendi: No valid embeddings')
                    results['vendi'] = None
                else:
                    embeddings = torch.stack(embeddings)
                    # Calculate Vendi Score
                    vendi_score = compute_vendi_score(embeddings)
                    
                    results['vendi'] = vendi_score
                    # Clean up memory
                    del embeddings, vendi_score
        finally:
            if 'vendi_model' in locals():
                del vendi_model
            clear_gpu_memory()
        print(f"Cond. Vendi: {results['vendi']:.4f}")
    
    # ====================================== All Results
    print("\n********************************")
    if args.result_path :
        if not os.path.isdir(args.result_path):
            Path(args.result_path).mkdir(parents=True, exist_ok=True)
        results_file_path = os.path.join(args.result_path, 'evaluation_results.txt')
        
    else:
        results_file_path = os.path.join(root_path, 'evaluation_results.txt')

    # Save individual scores to CSV
    save_all_dir = args.result_path or root_path
    score_dicts = {
        'clp_scores_dict_all': clp_scores_dict_all if 'clap' in args.modes else None,
        'kl_scores_dict': kl_scores_dict if 'kld' in args.modes else None,
    }
    if any(score_dict for score_dict in score_dicts.values() if score_dict):
        save_all_files_to_csv(result_path=save_all_dir, **score_dicts)
        print(f"Save individual scores csv Done: {save_all_dir}")


    with open(results_file_path, 'a', encoding='utf-8') as f:
        f.write(f'=== {args.dataset} Evaluation Results ===\n\n')
        f.write(f'Generated Audios Path: {results["root_path"]}\n\n')
        if 'clap' in args.modes:
            f.write(f'CLAP Score: {results["clap_score"]}\n')
        if 'kld' in args.modes:
            f.write(f'KLpasst: {results["klpasst"]}\n')
        if 'fd' in args.modes:
            f.write(f'FDopenl3: {results["fdopenl3"]}\n')
        if 'ipr' in args.modes:
            f.write(f'IPR-precision: {results["precision_mean"]}({results["precision_std"]})\n')
            f.write(f'IPR-recall: {results["recall_mean"]}({results["recall_std"]})\n')
        if 'vendi' in args.modes:
            f.write(f'Vendi: {results["vendi"]}\n')
        f.write('\n'+'-' * 50 + '\n')

        if results:
            f.write('================== Summary ================\n')
            #f.write(f'Total Experiments: {len(results)}\n')
            if 'clap' in args.modes:
                f.write(f'CLAP - Mean: {np.mean(results["clap_score_all"]):.4f}, Min: {np.min(results["clap_score_all"]):.4f}, Max: {np.max(results["clap_score_all"]):.4f}\n')
            if 'kld' in args.modes:
                f.write(f'KLpasst - Value: {results["klpasst"]:.4f}\n')
            if 'fd' in args.modes:
                f.write(f'FDopenl3 - Value: {results["fdopenl3"]:.4f}\n')
            if 'ipr' in args.modes:
                f.write(f'IPR Precision - Mean: {np.mean(results["precision_all"]):.4f}, Min: {np.min(results["precision_all"]):.4f}, Max: {np.max(results["precision_all"]):.4f}\n')
                f.write(f'IPR Recall - Mean: {np.mean(results["recall_all"]):.4f}, Min: {np.min(results["recall_all"]):.4f}, Max: {np.max(results["recall_all"]):.4f}\n')
                f.write(f'     basemodel: {args.clap_basemodel}({ipr_ref_options})\n')
                f.write(f'     basedata: {args.ipr_basedata}\n')
                f.write(f'     avg: {False}({ipr_num_avg})\n')
            if 'vendi' in args.modes:
                f.write(f'Vendi - Value: {results["vendi"]:.4f}\n')

    print(f"\nResults saved to {results_file_path}.")
    
    # Final memory cleanup and status output
    if torch.cuda.is_available():
        clear_gpu_memory()
    
    print("\n********************************")
    
    


if __name__ == '__main__':
    main()


