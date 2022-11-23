from datetime import datetime
import smtplib
from email.mime.text import MIMEText

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