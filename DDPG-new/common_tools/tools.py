from datetime import datetime
import smtplib
from email.mime.text import MIMEText
import os
import torch
import argparse

def get_common_args():

    parser = argparse.ArgumentParser()

    # resource
    parser.add_argument("--cpu_num", type=int, default=15, help='restrict the num of pytorch used CPU')
    parser.add_argument("--cuda", default=True, action='store_false')
    parser.add_argument("--gpu_id", type=str, default='0')

    # setting
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_dir", type=str, default='runs', help='path under os.getcwd()')
    parser.add_argument("--exp_postfix", type=str, default='', help='')
    parser.add_argument("--env_name", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--max_steps", type=int, default=int(10e7))
    parser.add_argument("--train_from_scratch", default=True, action='store_false')
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--seed", type=int, default=9999)

    # learning
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)

    # env
    parser.add_argument("--episode_life", default=True, action="store_false")
    parser.add_argument("--clip_rewards", default=True, action="store_false")

    return parser

def set_torch_cpu_num(args):
    if args.cpu_num != -1:
        os.environ['OMP_NUM_THREADS'] = str(args.cpu_num)
        os.environ['OPENBLAS_NUM_THREADS'] = str(args.cpu_num)
        os.environ['MKL_NUM_THREADS'] = str(args.cpu_num)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(args.cpu_num)
        os.environ['NUMEXPR_NUM_THREADS'] = str(args.cpu_num)
        torch.set_num_threads(args.cpu_num)

def common_setup(args):
    set_torch_cpu_num(args)
    if hasattr(args, 'max_steps'):
        args.save_freq = args.max_steps // 5  # save five models routinely
        args.eval_freq = max(args.max_steps // 1000, 1)  # eval 1000 times

    if args.debug:
        args.output_dir = 'runs/debug'
        args.max_steps = 1000
        args.train_freq = 5
        args.save_freq = 5
        args.eval_freq = 5

    args.output_dir += f'/{datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M")}'
    if hasattr(args, 'exp_postfix') and args.exp_postfix != '':
        args.output_dir += f'_{args.exp_postfix}'
    args.output_dir += f'/{args.env_name}'

    # postfix for common args
    if args.lr != 0.002:
        args.output_dir += f'_LR={args.lr}'
    if args.batch_size != 32:
        args.output_dir += f'_Batchsize={args.batch_size}'
    if hasattr(args, 'episode_life') and not args.episode_life:
        args.output_dir += f'_EpisodeLife=False'
    if hasattr(args, 'clip_rewards') and not args.clip_rewards:
        args.output_dir += f'_ClipRewards=False'
    if hasattr(args, 'train_from_scratch') and not args.train_from_scratch:
        args.output_dir += '_Xubei'

    return args

def send_an_error_message(program_name, error_name, error_detail):
    '''
    @program_name: 运行的程序名
    @error_name: 错误名
    @error_detail: 错误的详细信息
    @description: 程序出错是发送邮件提醒
    '''
    # SMTP 服务器配置
    SMTP_server = "smtp.qq.com"  # SMTP服务器地址 yyx:注意这里我使用的是QQ邮箱，所以要改成qq邮箱的SMTP
    email_address = "184035@qq.com"  # 邮箱地址
    Authorization_code = "oiwqnlxdfdmicaig"  # 授权码--用于登录第三方邮件客户端的专用密码，不是邮箱密码

    # 发件人和收件人
    sender = email_address  # 发件人，默认发件人等于email_address
    receivers = "184035@qq.com"  # 收件人

    # 获取程序出错的时间
    error_time = datetime.strftime(datetime.today(), "%Y-%m-%d %H:%M:%S:%f")
    # 邮件内容
    subject = "【yyx，您的程序又出bug了！】{name}-{date}".format(name=program_name, date=error_time)  # 邮件的标题
    content = '''<div class="emailcontent" style="width:100%;max-width:720px;text-align:left;margin:0 auto;padding-top:80px;padding-bottom:20px">
        <div class="emailtitle">
            <h1 style="color:#fff;background:#51a0e3;line-height:70px;font-size:24px;font-weight:400;padding-left:40px;margin:0">程序运行异常通知</h1>
            <div class="emailtext" style="background:#fff;padding:20px 32px 20px">
                <p style="color:#6e6e6e;font-size:13px;line-height:24px">程序：<span style="color:red;">【{program_name}】</span>运行过程中出现异常错误，下面是具体的异常信息，请及时核查处理！</p>
                <table cellpadding="0" cellspacing="0" border="0" style="width:100%;border-top:1px solid #eee;border-left:1px solid #eee;color:#6e6e6e;font-size:16px;font-weight:normal">
                    <thead>
                        <tr>
                            <th colspan="2" style="padding:10px 0;border-right:1px solid #eee;border-bottom:1px solid #eee;text-align:center;background:#f8f8f8">程序异常详细信息</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style="padding:10px 0;border-right:1px solid #eee;border-bottom:1px solid #eee;text-align:center;width:100px">异常简述</td>
                            <td style="padding:10px 20px 10px 30px;border-right:1px solid #eee;border-bottom:1px solid #eee;line-height:30px">{error_name}</td>
                        </tr>
                        <tr>
                            <td style="padding:10px 0;border-right:1px solid #eee;border-bottom:1px solid #eee;text-align:center">异常详情</td>
                            <td style="padding:10px 20px 10px 30px;border-right:1px solid #eee;border-bottom:1px solid #eee;line-height:30px">{error_detail}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
        '''.format(program_name=program_name, error_name=error_name, error_detail=error_detail)  # 邮件的正文部分
    # 实例化一个文本对象
    massage = MIMEText(content, 'html', 'utf-8')
    massage['Subject'] = subject  # 标题
    massage['From'] = sender  # 发件人
    massage['To'] = receivers  # 收件人

    try:
        mail = smtplib.SMTP_SSL(SMTP_server, 465)  # 连接SMTP服务，默认465和944 yyx这里使用465成功了，用994会报错
        mail.login(email_address, Authorization_code)  # 登录到SMTP服务
        mail.sendmail(sender, receivers, massage.as_string())  # 发送邮件
        print("成功发送了一封邮件到" + receivers)
    except smtplib.SMTPException as ex:
        print("邮件发送失败！")